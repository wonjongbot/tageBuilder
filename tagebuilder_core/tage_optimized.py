"""
TAGE Optimized Predictor Module

This module implements the TAGE (TAgged GEometric) branch predictor, including:
  - Global metadata and return type definitions.
  - History operations for global history updates.
  - Compressed history for reducing long global history into shorter forms.
  - Core logic for calculating predictions and updating predictors.
  - Tagged predictor structures and utility functions.
  - The TAGEPredictor class that ties together the predictor tables and metadata.
"""

import yaml
import numba
import numpy as np
from numba import prange
from numba import njit

### SECTION: GLOBALS
# Define numpy dtypes for metadata and return values
metadtype = np.dtype([
    ('base_idx_mask', np.uint32),
    ('base_pred_hyst_diff_log', np.uint32),
    ('phist', np.uint32),
    ('phist_len', np.uint32),
    ('use_alt_on_new_alloc', np.int8),
    ('u_tick', np.uint32),
    ('u_tick_log', np.uint32),
    ('ghist_head', np.uint32),
    ('ghist_ptr', np.uint32),
    ('ghist_size_log', np.uint32),
    ('rand', np.uint32)
    ])

retdtype = np.dtype([
    ('pid', np.int8),
    ('pred', np.int8),
    ('is_alt', np.int8),
    ('ctr', np.int8),
    ])

### SECTION: HISTORY OPERATIONS
@njit(inline='always')
def ghist_print(buf, meta, length = 64):
    """
    Print a segment of the global history buffer.
    
    Args:
        buf (np.array): Global history buffer.
        meta (np.array): Metadata array containing history pointers.
        length (int): Number of entries to print.
    """
    head = meta['ghist_head']
    # Compute cyclic indices for printing history
    idxs = (head + 1 + np.arange(length)) & ((1 << meta['ghist_size_log']) - 1)
    o = ''
    for it in idxs:
        o += str(buf[it])
    print(o)

@njit(inline='always')
def ghist_update(buf, val, meta):
    """
    Update the global history buffer with a new value.
    
    Args:
        buf (np.array): Global history buffer.
        val (int): New history bit to insert.
        meta (np.array): Metadata with history pointers.
    """
    head = meta['ghist_head']
    # Insert new value at the current head position
    buf[head] = val
    # Update the pointer for the most recent history value
    meta['ghist_ptr'] = head
    # Move the head pointer in the cyclic buffer
    meta['ghist_head'] = (head - 1) & ((1 << meta['ghist_size_log']) - 1)

@njit(inline='always')
def phist_update(bpc, meta):
    """
    Update the path history with a new branch PC bit.
    
    Args:
        bpc (int): Branch PC value.
        meta (np.array): Metadata containing the path history.
    """
    phist = meta['phist'] << 1 | (bpc & 1)
    # Keep phist within the defined bit-length
    meta['phist'] = phist & ((1 << meta['phist_len']) - 1)

### SECTION: COMPRESSED HISTORY
"""
Cyclic shift register functions to compress long global history into a shorter one,
used for table indexing and tagging.
"""
comp_hist_dtype = np.dtype([
    ('comp', np.uint32),
    ('orig_len', np.uint32),
    ('comp_len', np.uint32),
    ('offset', np.uint32),
    ('mask', np.uint32)
    ])

# TODO: strongly typed funciton for comp_hist
@njit(inline='always')
def comp_hist_init(comp_hist, orig_len:np.uint32, comp_len:np.uint32):
    """
    Initialize a compressed history structure.
    
    Args:
        comp_hist (np.array): Compressed history structure.
        orig_len (np.uint32): Original history length.
        comp_len (np.uint32): Compressed history length.
    """
    comp_hist['comp'] = 0
    comp_hist['orig_len'] = orig_len
    comp_hist['comp_len'] = comp_len
    # Calculate offset for the compression
    comp_hist['offset'] = orig_len % comp_len
    comp_hist['mask'] = (1 << comp_len) -1

@njit(inline='always')
def comp_hist_update(comp_hist, youngest, oldest):
    """
    Update the compressed history with the newest and oldest bits.
    
    Args:
        comp_hist (np.array): Compressed history structure.
        youngest (int): Most recent bit from history.
        oldest (int): Oldest bit from the relevant history window.
    """
    comp = (int(comp_hist['comp']) << 1 ) | youngest
    comp ^= (oldest << comp_hist['offset'])
    comp ^= (comp >> comp_hist['comp_len'])
    comp &= comp_hist['mask']

    comp_hist['comp'] = comp

### SECTION: CORE LOGIC
@njit(inline='always')
def mix_path_history(len, phist, pid, t_num_entries_log):
    """
    Mix the path history bits to generate a hash value.
    
    Args:
        len (int): History length.
        phist (int): Path history.
        pid (int): Predictor ID.
        t_num_entries_log (int): Log2 of number of tagged entries.
    
    Returns:
        np.uint32: Mixed hash value.
    """
    
    phist_masked = phist & ((1 << len) - 1)
    mask = (1 << t_num_entries_log) - 1

    A1 = phist_masked & mask
    A2 = phist_masked >> t_num_entries_log
    A2 = ((A2 << pid) & mask) + (A2 >> (t_num_entries_log - pid))

    phist_masked = A1^A2
    phist_masked = ((phist_masked << pid) & mask) + (phist_masked >> (t_num_entries_log - pid))
                            
    return np.uint32(phist_masked)

@njit(inline='always')
def get_tagged_idx(pid, bpc, t_num_entries_log, t_hist_len, t_comp_hist_idx, phist):
    """
    Compute the tagged index for a predictor table.
    
    Args:
        pid (int): Predictor ID.
        bpc (int): Branch PC.
        t_num_entries_log (int): Log2 of table entries.
        t_hist_len (int): Tagged history length.
        t_comp_hist_idx (int): Compressed history for index calculation.
        phist (int): Path history.
    
    Returns:
        int: Calculated index into the tagged table.
    """
    t_hist_len = min(16, t_hist_len)
    mask = (1 << t_num_entries_log) - 1
    foo = (bpc >> int(abs(int(t_num_entries_log) - pid) + 1))
    return (bpc ^ foo ^ t_comp_hist_idx ^ mix_path_history(t_hist_len, phist,pid, t_num_entries_log)) & mask

@njit(inline='always')
def get_tagged_tag(pid, bpc, t_comp_hist_tag0_comp, t_comp_hist_tag1_comp, t_tag_width):
    """
    Compute the tag for the tagged predictor entry.
    
    Args:
        pid (int): Predictor ID.
        bpc (int): Branch PC.
        t_comp_hist_tag0_comp (int): Compressed history component 0.
        t_comp_hist_tag1_comp (int): Compressed history component 1.
        t_tag_width (int): Width of the tag.
    
    Returns:
        int: Calculated tag.
    """
    mask = (1 << t_tag_width) - 1
    return (bpc ^ t_comp_hist_tag0_comp ^ (t_comp_hist_tag1_comp << 1)) & mask

@njit(inline='always')
def get_prediction(bpc, pid, b_entries, t_entries_num_log, t_entries, t_offset, t_idxs, t_tags,
                   t_hist_len, t_comp_hist_idx, t_comp_hist_tag0, t_comp_hist_tag1, t_tag_width, metadata):
    """
    Generate a branch prediction based on base and tagged predictor entries.
    
    Args:
        bpc (int): Branch PC.
        pid (int): Predictor ID.
        b_entries (np.array): Base predictor entries.
        t_entries_num_log (int): Log2 of tagged predictor entries.
        t_entries (np.array): Tagged predictor table.
        t_offset (int): Offset for tagged table access.
        t_idxs (np.array): Array for storing calculated indexes.
        t_tags (np.array): Array for storing calculated tags.
        t_hist_len (int): Tagged history length.
        t_comp_hist_idx (np.array): Compressed history for index calculation.
        t_comp_hist_tag0 (np.array): Compressed history component for tag.
        t_comp_hist_tag1 (np.array): Second compressed history component for tag.
        t_tag_width (int): Tag width.
        metadata (np.array): Global metadata.
    
    Returns:
        np.array: Array [prediction, is_hit, pred_counter].
    """
    # Base prediction when pid == 0 (bimodal predictor)
    if pid == 0:
        idx_bim = (bpc) & metadata['base_idx_mask']
        prediction_entry = b_entries[idx_bim]
        hyst_entry = b_entries[idx_bim >> metadata['base_pred_hyst_diff_log']]
        ctr = (b_entries[idx_bim] & 0b10) | (hyst_entry & 0b1)
        #print(branch_pc, base_entries[idx_bim])
        return np.array([(prediction_entry >> 1) & 0b1, 1, ctr], dtype=np.int8)
    # Tagged predictions
    else:
        phist = metadata['phist']
        t_idx = get_tagged_idx(pid, bpc, t_entries_num_log, t_hist_len, t_comp_hist_idx['comp'], phist)
        t_tag = get_tagged_tag(pid, bpc, t_comp_hist_tag0['comp'], t_comp_hist_tag1['comp'], t_tag_width)
        t_idxs[pid] = t_idx
        t_tags[pid] = t_tag
        e = t_entries[t_offset + t_idx]
        # Check if tag matches
        if e['tag'] == t_tag:
            return np.array([e['pred_ctr'] >= 0, 1, e['pred_ctr']], dtype=np.int8)
        else:
            return np.array([0, 0, -1], dtype=np.int8)

@njit(inline='always')
def update_base_predictor(branch_pc, branch_taken, base_entries, metadata):
    """
    Update the base predictor counters.
    
    Args:
        branch_pc (int): Branch PC.
        branch_taken (bool): Outcome of the branch.
        base_entries (np.array): Base predictor table.
        metadata (np.array): Global metadata.
    
    Returns:
        int: Always returns 0.
    """
    # base predictor update
    diff = metadata['base_pred_hyst_diff_log']
    idx_bim = (branch_pc) & metadata['base_idx_mask']
    hyst_ctr = base_entries[idx_bim >> diff]
    pred_ctr = base_entries[idx_bim]

    # Combine hysteresis and prediction counters
    ctr = (hyst_ctr << 1) & 0b10
    ctr |= (pred_ctr & 0b1)
    
    # Adjust counter based on branch outcome
    if branch_taken:
        updated_ctr = min(ctr + 1, 3)
    else:
        updated_ctr = max(ctr - 1, 0)

    # Update hysteresis and prediction entries
    base_entries[idx_bim >> diff] = (hyst_ctr & 0b10) | (updated_ctr & 0b01)
    base_entries[idx_bim] = (updated_ctr & 0b10) | (pred_ctr & 0b01)
    return 0

@njit(inline='always')
def update_tagged_ctr(branch_taken, tagged_entry):
    """
    Update the counter for a tagged predictor entry.
    
    Args:
        branch_taken (bool): Outcome of the branch.
        tagged_entry (np.array): Tagged predictor entry.
    """
    oldval = tagged_entry['pred_ctr']
    if branch_taken:
        tagged_entry['pred_ctr'] = min(oldval+1, 3)
    else:
        tagged_entry['pred_ctr'] = max(oldval-1, -4)


@njit(inline='always')
def make_pred_n_update_batch(br_infos, num_tables, base_entries, tagged_entries, tagged_offsets, 
                             tagged_idxs, tagged_tags, tagged_num_entries_log, hist_len_arr, 
                             comp_hist_idx_arr, comp_hist_tag0_arr, comp_hist_tag1_arr, 
                             tagged_tag_widths, ghist, metadata, rand_array):
    """
    Process a batch of branch instructions to make predictions and update predictor tables.
    
    Args:
        br_infos (np.array): Array of branch information.
        num_tables (int): Total number of predictor tables.
        base_entries (np.array): Base predictor table.
        tagged_entries (np.array): Combined tagged predictor entries.
        tagged_offsets (np.array): Offsets for each tagged table.
        tagged_idxs (np.array): Array to store calculated indices.
        tagged_tags (np.array): Array to store calculated tags.
        tagged_num_entries_log (np.array): Log2 sizes of tagged tables.
        hist_len_arr (np.array): History lengths for each table.
        comp_hist_idx_arr (np.array): Compressed history for index calculation per table.
        comp_hist_tag0_arr (np.array): Compressed history for tag component 0.
        comp_hist_tag1_arr (np.array): Compressed history for tag component 1.
        tagged_tag_widths (np.array): Tag widths for each table.
        ghist (np.array): Global history buffer.
        metadata (np.array): Global metadata.
        rand_array (np.array): Array of random values.
    
    Returns:
        np.array: Array of prediction results containing:
                  - pid: Predictor ID used.
                  - pred: Final prediction (1 for taken, 0 for not taken).
                  - is_alt: Flag indicating if an alternate predictor was used.
                  - ctr: Confidence counter.
    """
    results = np.zeros(len(br_infos), dtype=retdtype)
    for i, b in enumerate(br_infos):
        # Initialize predictions for each table
        predictions = np.zeros((num_tables, 3), dtype=np.int8)
        for predictor_id in range(num_tables): #prange(num_tables):
            predictions[predictor_id, :] = get_prediction(
                b['addr'],
                predictor_id,
                base_entries,
                tagged_num_entries_log[predictor_id],
                tagged_entries,
                tagged_offsets[predictor_id],
                tagged_idxs,
                tagged_tags,
                hist_len_arr[predictor_id],
                comp_hist_idx_arr[predictor_id],
                comp_hist_tag0_arr[predictor_id],
                comp_hist_tag1_arr[predictor_id],
                tagged_tag_widths[predictor_id],
                metadata
            )
        # Determine which predictors had a hit (tag match)
        boolean_mask = (predictions[:,1] == 1)
        indices = np.where(boolean_mask)[0]

        if len(indices) >= 2:
            main_pred_id = indices[-1]          # Last hit becomes main prediction
            alt_pred_id = indices[-2]           # Second-to-last becomes alternate
            main_pred = predictions[main_pred_id, 0]
            alt_pred = predictions[alt_pred_id, 0]
        elif len(indices) == 1:
            # If only one True value exists
            main_pred_id = indices[-1]
            alt_pred_id = indices[-1]
            main_pred = predictions[main_pred_id, 0]
            alt_pred = predictions[main_pred_id, 0]
        else:
           # Should not occur; ensure at least one hit
            main_pred_id = None
            alt_pred_id = None
            main_pred = None
            alt_pred = None
            assert(False)

        # Select final prediction based on predictor counters and allocation policy
        if main_pred_id > 0:
            if (metadata['use_alt_on_new_alloc'] < 0) or (predictions[int(main_pred_id), 2] not in (-1, 0)):
                final_pred = main_pred
                is_alt = False
                final_pred_id = main_pred_id
            else:
                final_pred = alt_pred
                is_alt = True
                final_pred_id = alt_pred_id
        else:
            final_pred = alt_pred
            is_alt = True
            final_pred_id = alt_pred_id

        # Save prediction result
        results[i]['pid'] = final_pred_id
        results[i]['pred'] = final_pred
        results[i]['is_alt'] = is_alt
        results[i]['ctr'] = predictions[final_pred_id, 2]
        isCorrect = np.uint8(final_pred == b['taken'])

        # UPDATE PREDICTORS TODO: Move this into a seperate function
        # Update predictors based on outcome
        rand_val = rand_array[metadata['rand']]
        metadata['rand'] = (metadata['rand'] + 1) % 10000
        is_new_alloc = (not isCorrect) and (main_pred_id < num_tables - 1)
        
        # Allocate new table entry for update
        if main_pred_id > 0:
            # update weak results
            pseudo_new_alloc = predictions[int(main_pred_id), 2] in (-1,0) 
            if pseudo_new_alloc:
                if main_pred == b['taken']:
                    is_new_alloc = False
                if main_pred != alt_pred:
                    if alt_pred == b['taken']:
                        if metadata['use_alt_on_new_alloc'] < 7:
                            metadata['use_alt_on_new_alloc'] += 1
                    elif metadata['use_alt_on_new_alloc'] > -8:
                        metadata['use_alt_on_new_alloc'] -= 1
                if metadata['use_alt_on_new_alloc'] >= 0:
                    # set final pred to main predictor for useful bit update
                    final_pred = main_pred

        if is_new_alloc:
            # pick a new entry to allocate
            num_table_avail = num_tables - 1 - main_pred_id
            num_table_avail = min(3, num_table_avail)
            random_offset = rand_val % num_table_avail
            new_entry_id  = main_pred_id + 1 + random_offset
            
            min_u = 1
            u_values = np.array([
                tagged_entries[tagged_offsets[j]+tagged_idxs[j]]['u'] for j in range(main_pred_id+1, num_tables)
            ])
            min_u = np.min(u_values)
            
            # clear out existing entry if none availble
            if min_u > 0:
                tagged_entries[tagged_offsets[new_entry_id]+tagged_idxs[new_entry_id]]['u'] = np.uint8(0)
            
            for j in range(new_entry_id, num_tables): #NOTE: fix math?
                entry = tagged_entries[tagged_offsets[j]+tagged_idxs[j]]
                if entry['u'] == 0:
                    entry['tag'] = tagged_tags[j]
                    entry['pred_ctr'] = np.int8(0) if b['taken'] else np.int8(-1)
                    break
        # Allocate done
        
        # reset useful bit upon saturation NOTE: ANY WAY TO OPTIMIZE THIS??
        metadata['u_tick'] += 1
        if (metadata['u_tick'] & ((1 << metadata['u_tick_log']) - 1)) == 0:
            for j in range(tagged_entries.shape[0]):
                tagged_entries[j]['u'] = tagged_entries[j]['u'] >> 1
        # reset ubit done

        # Update counters based on outcome and useful bit
        if main_pred_id > 0:
            offset = tagged_offsets[main_pred_id]
            idx = tagged_idxs[main_pred_id]
            entry_new = tagged_entries[offset + idx]
            update_tagged_ctr(b['taken'], entry_new)
            
            if entry_new['u'] == 0:
                if alt_pred_id > 0:
                    alt_entry = tagged_entries[tagged_offsets[alt_pred_id]+tagged_idxs[alt_pred_id]]
                    update_tagged_ctr(b['taken'], alt_entry)
                elif alt_pred_id == 0:
                    update_base_predictor(b['addr'], b['taken'], base_entries, metadata)
        else:
            update_base_predictor(b['addr'], b['taken'], base_entries, metadata)
            
        # update usefulness bit based on outcome
        if final_pred != alt_pred:
            if final_pred == b['taken']:
                entry_new['u'] = min(int(entry_new['u']) + 1, 3)
                #print(entry_new['u'])
            else:
                if metadata['use_alt_on_new_alloc'] < 0:
                    entry_new['u'] = max(int(entry_new['u']) - 1, 0)
        
        # update global and path history
        ghist_update(ghist, b['taken'], metadata)
        phist_update(b['addr'], metadata)

        # update compressed histories for all tagged tables
        ghist_mask = (1 << metadata['ghist_size_log']) - 1
        for id in range(1, num_tables):
            oldestIdx = hist_len_arr[id]
            youngest_val = ghist[(1 + metadata['ghist_head']) & ghist_mask]
            oldest_val = ghist[(1 + metadata['ghist_head'] + oldestIdx) & ghist_mask]

            comp_hist_update(comp_hist_idx_arr[id], youngest_val, oldest_val)
            comp_hist_update(comp_hist_tag0_arr[id], youngest_val, oldest_val)
            comp_hist_update(comp_hist_tag1_arr[id], youngest_val, oldest_val)

    
    return results

### SECTION: TAGGED PREDICTOR ENTRY
"""
datatype for each entry in tagged predictors
"""
tagged_entry_dtype = np.dtype([
    ('tag', np.uint32),
    ('pred_ctr', np.int8),
    ('u', np.uint8)
    ]) 

### SECTION: UTILITIES
def get_entry(predictor, id, idx):
    """
    Retrieve a predictor entry from either the base or tagged predictor table.
    Used for debug purposes
    
    Args:
        predictor (TAGEPredictor): Instance of the predictor.
        id (int): Table id (0 for base predictor).
        idx (int): Index into the table.
    
    Returns:
        The predictor entry at the given index.
    """
    if id == 0:
        return predictor.base_entries[idx]
    else:
        return predictor.tagged_entries[predictor.tagged_offsets[id]+idx]
### UTILITIES DONE


### SECTION: TAGEPredictor Class
"""
TAGEPredictor class holds predictor tables and metadatas 
core logics are moved to function for optimization purposes
"""
# core logic are moved to function for optimization purposes
class TAGEPredictor:
    """
    Class that implements the TAGE branch predictor.
    
    This class holds architectural information of the predictor, 
    including predictor tables and associated metadata,
    """
    def __init__(self, spec, logger):
        """
        Initialize the TAGEPredictor with the given specification.
        
        Args:
            spec (dict): Specification dictionary defining tables and global configurations.
            logger: Logger instance for logging information.
        """
        self.storage_report = {}

        self.logger = logger

        self.num_tables = len(spec['tables'])
        self.num_tot_tagged_entries = 0
        self.id2name = []
        self.max_hist_len = np.uint32(0)

        # Initialize random number generator and array for allocation randomness.
        self.rng = np.random.default_rng()
        self.rand_array = self.rng.integers(low=0, high=3, size=10000)

        # Predictor tables: base and tagged.
        self.base_entries = None
        self.tagged_entries = None

        # Base predictor attributes.
        self.base_num_pred_entries_log = None
        self.base_num_hyst_entries_log = None

        # Tagged predictor attributes.
        self.tagged_tag_widths = np.zeros(self.num_tables, dtype=np.uint32)
        self.tagged_num_entries_log = np.zeros(self.num_tables, dtype=np.uint32)
        self.tagged_idxs = np.zeros(self.num_tables, dtype=np.uint32)
        self.tagged_tags = np.zeros(self.num_tables, dtype=np.uint32)
        self.tagged_offsets = np.zeros(self.num_tables, dtype=np.uint32) # offsets for accessing one large table

        self.hist_len_arr = np.zeros(self.num_tables, dtype=np.uint32)
        self.comp_hist_idx_arr = np.zeros(self.num_tables, dtype = comp_hist_dtype)
        self.comp_hist_tag0_arr = np.zeros(self.num_tables, dtype = comp_hist_dtype)
        self.comp_hist_tag1_arr = np.zeros(self.num_tables, dtype = comp_hist_dtype)

        self.metadata = np.zeros(1, dtype=metadtype)
        
        # Initialize each table based on the specification.
        for id, table in enumerate(spec['tables']):
            self.id2name.append(table['name'])
            table_size = 2**table['num_pred_entries_log']
            self.hist_len_arr[id] = table['hist_len']

            if table['isBase']:
                # Set up base predictor metadata and table.
                self.metadata[0]['base_idx_mask'] = ((1 << table['num_pred_entries_log']) - 1)
                self.base_entries = np.zeros(table_size, dtype = np.int8)
                self.base_num_pred_entries_log = table['num_pred_entries_log']
                self.base_num_hyst_entries_log = table['num_hyst_entries_log']
                self.metadata[0]['base_pred_hyst_diff_log'] = self.base_num_pred_entries_log - self.base_num_hyst_entries_log
            else:
                self.num_tot_tagged_entries += table_size
                # Compute offset for this table in the combined tagged table.
                for i in range(id + 1, self.num_tables):
                    self.tagged_offsets[i] += table_size
                self.tagged_num_entries_log[id] = table['num_pred_entries_log']
                # Initialize compressed history for the tagged table.
                self.tagged_tag_widths[id] = table['tag_width'] 
                comp_hist_init(self.comp_hist_idx_arr[id], self.hist_len_arr[id], table['num_pred_entries_log'])
                comp_hist_init(self.comp_hist_tag0_arr[id], self.hist_len_arr[id], self.tagged_tag_widths[id])
                comp_hist_init(self.comp_hist_tag1_arr[id], self.hist_len_arr[id], self.tagged_tag_widths[id] - 1) # to distribute tag hash further
        
        # Create the combined tagged predictor table.
        self.tagged_entries = np.zeros(self.num_tot_tagged_entries, dtype=tagged_entry_dtype)
        self.max_hist_len = max(self.hist_len_arr)

        # TODO: Add seperate history registers for system calls
        # Set up global history buffer size.
        if self.max_hist_len > 0:
            ghist_size_log = int(np.ceil(np.log2(self.max_hist_len)))
        else:
            ghist_size_log = 1
        self.ghist_bufsize = 2**ghist_size_log
        self.ghist = np.zeros(self.ghist_bufsize ,dtype = np.uint8)

        # Initialize additional metadata.
        self.metadata[0]['ghist_size_log'] = ghist_size_log
        self.metadata[0]['phist_len'] = spec['global_config']['phist_len']
        self.metadata[0]['u_tick_log'] = spec['global_config']['u_duration_log']

        # Log configuration details.
        self.logger.info(f'hist_len_arr:\n    {self.hist_len_arr}')
        self.logger.info(f'comp_hist_idx:\n    {self.comp_hist_idx_arr}')
        self.logger.info(f'comp_hist_tag0:\n   {self.comp_hist_tag0_arr}')
        self.logger.info(f'comp_hist_tag1:\n   {self.comp_hist_tag1_arr}')
        self.logger.info(f'tagged_offsets:\n    {self.tagged_offsets}')
        self.logger.info(f'base_size:\n    {len(self.base_entries)}')
        self.logger.info(f'tagged_size:\n    {len(self.tagged_entries)}')
        self.logger.info(f'tag_widths:\n    {self.tagged_tag_widths}')

        # Report storage information.
        self.storage_report['ghist_size_b'] = self.max_hist_len
        self.storage_report['phist_size_b'] = self.metadata[0]['phist_len']
        self.storage_report['use_alt_on_new_alloc'] = 4
        self.storage_report['base'] = {}
        self.storage_report['base']['pred_bit_b'] = 2**self.base_num_pred_entries_log
        self.storage_report['base']['pred_hyst_b'] = 2**self.base_num_hyst_entries_log
        self.storage_report['tagged'] = {}
        for id in range(1, self.num_tables):
            if id < self.num_tables - 1:
                self.storage_report['tagged'][f'{id}_b'] = (3 + 2 + self.tagged_tag_widths[id])*(self.tagged_offsets[id + 1] - self.tagged_offsets[id])
            else:
                self.storage_report['tagged'][f'{id}_b'] = (3 + 2 + self.tagged_tag_widths[id])*(len(self.tagged_entries) - self.tagged_offsets[id])
        
        total_storage = 0
        for k,v in self.storage_report.items():
            if type(v) == type({}):
                for k1,v1 in v.items():
                    total_storage += v1
            else:
                total_storage += v

        self.storage_report['tot_size_b'] = total_storage
        
        self.logger.info("storage_info:")
        for k,v in self.storage_report.items():
            self.logger.info(f"    {k}: {v}")
