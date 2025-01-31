import yaml
import numba
import numpy as np
from numba import prange
from numba import njit

### SECTINO: GLOBALS
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
### GLOBAL DONE

### SECTION: HISTORY OPERATIONS
@njit(inline='always')
def ghist_print(buf, meta, length = 64):
        """
        return a np array with 0th element being head and <length - 1>th
        entry being last element of the array
        """
        head = meta['ghist_head']
        idxs = (head + 1 + np.arange(length)) & ((1 << meta['ghist_size_log']) - 1)
        
        o = ''
        for it in idxs:
            o += str(buf[it])
        print(o)

@njit(inline='always')
def ghist_update(buf, val, meta):
    head = meta['ghist_head']
    # insert new history value to head (next available spot in cyclic buffer)
    buf[head] = val
    # update ptr (most current value) position
    meta['ghist_ptr'] = head
    # update head position
    meta['ghist_head'] = (head - 1) & ((1 << meta['ghist_size_log']) - 1)
    #print(meta['ghist_head'], 1 << meta['ghist_size_log'] - 1)

@njit(inline='always')
def phist_update(bpc, meta):
    phist = meta['phist'] << 1 | (bpc & 1)
    meta['phist'] = phist & ((1 << meta['phist_len']) - 1)
### HISTORY OPERATIONS

### SECTION: COMPRESSED HISTORY
"""
cyclic shift register to compress long global history into
shorter one for table indexing and tagging.
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
    comp_hist['comp'] = 0
    comp_hist['orig_len'] = orig_len
    comp_hist['comp_len'] = comp_len
    comp_hist['offset'] = orig_len % comp_len
    #print(comp_len)
    comp_hist['mask'] = (1 << comp_len) -1

@njit(inline='always')
def comp_hist_update(comp_hist, youngest, oldest):
    comp = (int(comp_hist['comp']) << 1 ) | youngest
    comp ^= (oldest << comp_hist['offset'])
    comp ^= (comp >> comp_hist['comp_len'])
    comp &= comp_hist['mask']

    comp_hist['comp'] = comp
### COMPRESSED HISTORY DONE


### SECTION: CORE LOGIC
@njit(inline='always')
def mix_path_history(
    len,
    phist,
    pid,
    t_num_entries_log
    ):
    """
    replicates the `F` function in seznec's code
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
def get_tagged_idx(
    pid,
    bpc,
    t_num_entries_log, # log of number of entries 
    t_hist_len, # tagged history len
    t_comp_hist_idx, # compressed history for idx calculation
    phist, # phist
    ):
    t_hist_len = min(16, t_hist_len)
    mask = (1 << t_num_entries_log) - 1
    foo = (bpc >> int(abs(int(t_num_entries_log) - pid) + 1))
    return (bpc ^ foo ^ t_comp_hist_idx ^ mix_path_history(t_hist_len, phist,pid, t_num_entries_log)) & mask

@njit(inline='always')
def get_tagged_tag(
    pid,
    bpc,
    t_comp_hist_tag0_comp,
    t_comp_hist_tag1_comp,
    t_tag_width
    ):
    mask = (1 << t_tag_width) - 1
    return (bpc ^ t_comp_hist_tag0_comp ^ (t_comp_hist_tag1_comp << 1)) & mask

@njit(inline='always')
def get_prediction(
    bpc, # branch pc
    pid, # predictor id
    b_entries, # base entries array
    t_entries_num_log,
    t_entries, # tagged entries array
    t_offset, # tagged access offset
    t_idxs,
    t_tags,
    t_hist_len, # tagged predictor history lengths
    t_comp_hist_idx, # tagged predictor compressed hist
                     #  for idx calculation
    t_comp_hist_tag0,
    t_comp_hist_tag1,
    t_tag_width,
    #phist,
    metadata
    ): # returns NUMPY ARR
    """
    return value: numpy arr (prediction, is_hit, pred_counter)
    """
    # base prediction
    if pid == 0:
        idx_bim = (bpc) & metadata['base_idx_mask']
        #print(branch_pc, base_entries[idx_bim])
        return np.array([np.int8((b_entries[idx_bim] >> 1) & 0b1), np.int8(1), b_entries[idx_bim]], dtype=np.int8)
    # tage prediction (single predictor)
    else:
        phist = metadata['phist']
        t_idx = get_tagged_idx(
            pid,
            bpc,
            t_entries_num_log,
            t_hist_len,
            t_comp_hist_idx['comp'],
            phist
        )
        t_tag = get_tagged_tag(
            pid,
            bpc,
            t_comp_hist_tag0['comp'],
            t_comp_hist_tag1['comp'],
            t_tag_width
        )
        t_idxs[pid] = t_idx
        t_tags[pid] = t_tag
        e = t_entries[t_offset + t_idx]
        # NOTE: is it possible to somehow notify parent function
        #       so that we don't have to predict from unecessary ones?
        if e['tag'] == t_tag:
            return np.array([np.int8(e['pred_ctr'] >= 0), np.int8(1), e['pred_ctr']], dtype=np.int8)
        else:
            return np.array([np.int8(0), np.int8(0), np.int8(-1)], dtype=np.int8)

@njit(inline='always')
def update_base_predictor(
    branch_pc,
    branch_taken,
    # predictor_id,
    base_entries,
    # tagged_entries,
    # pred_results,
    # main_pred_id,
    # alt_pred_id,
    # t_idxs,
    # t_tags,
    # ghist,
    # phist,
    # comp_hist_idx_arr,
    # comp_hist_tag0_arr,
    # comp_hist_tag1_arr,
    metadata
    ):
    # base predictor update
    diff = metadata['base_pred_hyst_diff_log']
    idx_bim = (branch_pc) & metadata['base_idx_mask']
    hyst_ctr = base_entries[idx_bim >> diff]
    pred_ctr = base_entries[idx_bim]

    ctr = (hyst_ctr << 1) & 0b10
    ctr |= (pred_ctr & 0b1)

    if branch_taken:
        updated_ctr = min(ctr + 1, 3)
    else:
        updated_ctr = max(ctr - 1, 0)

    # update hysterisis bit
    base_entries[idx_bim >> diff] = (hyst_ctr & 0b10) | (updated_ctr & 0b01)
    # update prediction bit
    base_entries[idx_bim] = (updated_ctr & 0b10) | (pred_ctr & 0b01)
    # base predictor update done
    return 0

@njit(inline='always')
def update_tagged_ctr(branch_taken, tagged_entry):
    oldval = tagged_entry['pred_ctr']
    if branch_taken:
        tagged_entry['pred_ctr'] = min(oldval+1, 3)
    else:
        tagged_entry['pred_ctr'] = max(oldval-1, -4)


@njit(inline='always')
def make_pred_n_update_batch(
    br_infos,
    num_tables,
    base_entries,
    tagged_entries,
    tagged_offsets,
    tagged_idxs,
    tagged_tags,
    tagged_num_entries_log,
    hist_len_arr,
    comp_hist_idx_arr,
    comp_hist_tag0_arr,
    comp_hist_tag1_arr,
    tagged_tag_widths,
    ghist,
    #phist,
    #use_alt_on_new_alloc,
    metadata,
    rand_array
    ):
    """
    return 0 if prediction is false 1 if true : numpy array uint8
    """
    #predictor_id = 0
    results = np.zeros(len(br_infos), dtype=retdtype)
    for i, b in enumerate(br_infos):
        # any way to keep parallelized pools? need to reduce 
        #   parallellization overhead
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
                #phist,
                metadata
            )
            #print(predictions)
        #print(tagged_idxs)
        #print(tagged_tags)
        # use boolean mask to flag any hits
        boolean_mask = (predictions[:,1] == 1)
        
        # extract pred indicies with hits
        indices = np.where(boolean_mask)[0]

        if len(indices) >= 2:
            main_pred_id = indices[-1]        # Index of last True
            alt_pred_id = indices[-2] # Index of second-to-last True
            main_pred = predictions[main_pred_id, 0]
            alt_pred = predictions[alt_pred_id, 0]
        elif len(indices) == 1:
            # If only one True value exists
            main_pred_id = indices[-1]
            alt_pred_id = indices[-1]
            main_pred = predictions[main_pred_id, 0]
            alt_pred = predictions[main_pred_id, 0]
        else:
            # If no True values exist (shouldn't happen)
            main_pred_id = None
            alt_pred_id = None
            main_pred = None
            alt_pred = None
            assert(False)

        # if None in (main_pred_id, alt_pred_id, main_pred, alt_pred):
        #     assert(False)
        # CHANGE RETURN VALUE SO IT GIVES COUNTER ITSELF??
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

        #print(pred, b['taken'])
        
        # DEBUG
        # print('table id used:',main_pred_id)
        # results[i] = (final_pred_id, final_pred, is_alt, predictions[final_pred_id, 2])
        results[i]['pid'] = final_pred_id
        results[i]['pred'] = final_pred
        results[i]['is_alt'] = is_alt
        results[i]['ctr'] = predictions[final_pred_id, 2]
        isCorrect = np.uint8(final_pred == b['taken'])

        # UPDATE PREDICTORS

        rand_val = rand_array[metadata['rand']]
        metadata['rand'] = (metadata['rand'] + 1) % 10000
        #if (main_pred_id == num_tables - 1):
        #    print("using largest ghist for id", main_pred_id)
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
            min_u = 1
            u_values = np.array([
                tagged_entries[tagged_offsets[j]+tagged_idxs[j]]['u'] for j in range(main_pred_id+1, num_tables)
                ])
            min_u = np.min(u_values)

            # find entry to allocate
            num_table_avail = num_tables - 1 - main_pred_id
            num_table_avail = min(3, num_table_avail)
            random_offset = rand_val % num_table_avail
            new_entry_id  = main_pred_id + 1 + random_offset

            # clear out existing entry if none availble
            if min_u > 0:
                tagged_entries[tagged_offsets[new_entry_id]+tagged_idxs[new_entry_id]]['u'] = np.uint8(0)
            
            for j in range(new_entry_id, num_tables): #NOTE: fix math?
                entry = tagged_entries[tagged_offsets[j]+tagged_idxs[j]]
                if entry['u'] == 0:
                    entry['tag'] = tagged_tags[j]
                    entry['pred_ctr'] = np.int8(0) if b['taken'] else np.int8(-1)
                    # entry['u'] = np.uint8(0) # redundent 
                    # DEBUG
                    # print('entry to be inserted', j, entry)
                    break
        # Allocate done
        
        # reset useful bit upon saturation NOTE: ANY WAY TO OPTIMIZE THIS??
        metadata['u_tick'] += 1
        if (metadata['u_tick'] & ((1 << metadata['u_tick_log']) - 1)) == 0:
            for j in range(tagged_entries.shape[0]):
                tagged_entries[j]['u'] = tagged_entries[j]['u'] >> 1
        # reset ubit done

        # Update ctrs
        #entry_new = None
        #print(main_pred_id)
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
                    #assert(alt_pred_id == 0)
                    update_base_predictor(
                        b['addr'], b['taken'],
                        base_entries, metadata
                    )
        else:
            update_base_predictor(
                b['addr'], b['taken'],
                base_entries, metadata
            )
        # update usefulness
        if final_pred != alt_pred:
            if final_pred == b['taken']:
                entry_new['u'] = min(int(entry_new['u']) + 1, 3)
                #print(entry_new['u'])
            else:
                if metadata['use_alt_on_new_alloc'] < 0:
                    entry_new['u'] = max(int(entry_new['u']) - 1, 0)
        
        # update ghist and phist
        #print(b['taken'], ghist)
        ghist_update(ghist, b['taken'], metadata)
        #print('BRANCH:', b['taken'])
        #ghist_print(ghist,metadata,64)
        #print()
        phist_update(b['addr'], metadata)

        # update compressed histories
        ghist_mask = (1 << metadata['ghist_size_log']) - 1
        for id in range(1, num_tables):
            oldestIdx = hist_len_arr[id]
            youngest_val = ghist[(1 + metadata['ghist_head']) & ghist_mask]
            oldest_val = ghist[(1 + metadata['ghist_head'] + oldestIdx) & ghist_mask]

            comp_hist_update(comp_hist_idx_arr[id], youngest_val, oldest_val)
            comp_hist_update(comp_hist_tag0_arr[id], youngest_val, oldest_val)
            comp_hist_update(comp_hist_tag1_arr[id], youngest_val, oldest_val)

    
    #print(results)
    return results



### BIMODAL PREDICTOR DONE

### SECTION: TAGGED PREDICTOR
"""
datatype for each entry in tagged predictors
"""
tagged_entry_dtype = np.dtype([
    ('tag', np.uint32),
    ('pred_ctr', np.int8),
    ('u', np.uint8)
    ]) 
### TAGGED PREDICTOR DONE


### SECTION: UTILITIES
def get_entry(predictor, id, idx):
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
    def __init__(self, spec, logger):
        self.storage_report = {}

        self.logger = logger

        self.num_tables = len(spec['tables'])
        self.num_tot_tagged_entries = 0
        self.id2name = []
        self.max_hist_len = np.uint32(0)

        # randoms
        self.rng = np.random.default_rng()
        self.rand_array = self.rng.integers(low=0, high=3, size=10000)

        # DEPRECATED NOTE: consider combining these lists into a single large buffer
        # self.pred_ctr_list = numba.typed.List()
        # self.tag_list = numba.typed.List()
        # self.u_list = numba.typed.List()

        # table entries
        self.base_entries = None
        self.tagged_entries = None

        # base predictor attributes
        self.base_num_pred_entries_log = None
        self.base_num_hyst_entries_log = None

        # tagged predictor attributes
        self.tagged_tag_widths = np.zeros(self.num_tables, dtype=np.uint32)
        self.tagged_num_entries_log = np.zeros(self.num_tables, dtype=np.uint32)
        self.tagged_idxs = np.zeros(self.num_tables, dtype=np.uint32)
        self.tagged_tags = np.zeros(self.num_tables, dtype=np.uint32)
        # offsets for accessing one large table
        self.tagged_offsets = np.zeros(self.num_tables, dtype=np.uint32)

        self.hist_len_arr = np.zeros(self.num_tables, dtype=np.uint32)
        self.comp_hist_idx_arr = np.zeros(self.num_tables, dtype = comp_hist_dtype)
        self.comp_hist_tag0_arr = np.zeros(self.num_tables, dtype = comp_hist_dtype)
        self.comp_hist_tag1_arr = np.zeros(self.num_tables, dtype = comp_hist_dtype)

        #self.use_alt_on_new_alloc = np.int8(0)
        self.metadata = np.zeros(1, dtype=metadtype)

        for id, table in enumerate(spec['tables']):
            self.id2name.append(table['name'])
            table_size = 2**table['num_pred_entries_log']
            # saturating counter logic (3 bit counter):
            # [-3, 0]: not taken
            # [1, 4]:   taken
            # self.pred_ctr_list.append(np.zeros(table_size, dtype = np.int8))
            self.hist_len_arr[id] = table['hist_len']

            if table['isBase']:
                #global BASE_IDX_OFFSET
                self.metadata[0]['base_idx_mask'] = ((1 << table['num_pred_entries_log']) - 1)
                
                self.base_entries = np.zeros(table_size, dtype = np.int8)
                self.base_num_pred_entries_log = table['num_pred_entries_log']
                self.base_num_hyst_entries_log = table['num_hyst_entries_log']
                self.metadata[0]['base_pred_hyst_diff_log'] = self.base_num_pred_entries_log - self.base_num_hyst_entries_log
                # self.tag_list.append(np.zeros(0, dtype = np.uint16))    
                # self.u_list.append(np.zeros(0, dtype = np.uint8))
            else:
                self.num_tot_tagged_entries += table_size
                # offset calculation for future use
                for i in range(id + 1, self.num_tables):
                    self.tagged_offsets[i] += table_size
                self.tagged_num_entries_log[id] = table['num_pred_entries_log']
                #self.tag_list.append(np.zeros(table_size, dtype = np.uint16))    
                #self.u_list.append(np.zeros(table_size, dtype = np.uint8))
                # initialize compressed history attributes
                self.tagged_tag_widths[id] = table['tag_width'] 
                comp_hist_init(self.comp_hist_idx_arr[id], self.hist_len_arr[id], table['num_pred_entries_log'])
                comp_hist_init(self.comp_hist_tag0_arr[id], self.hist_len_arr[id], self.tagged_tag_widths[id])
                comp_hist_init(self.comp_hist_tag1_arr[id], self.hist_len_arr[id], self.tagged_tag_widths[id] - 1) # to distribute tag hash further
        self.tagged_entries = np.zeros(self.num_tot_tagged_entries, dtype=tagged_entry_dtype)
        self.max_hist_len = max(self.hist_len_arr)

        # TODO: Add seperate history registers for system calls
        if self.max_hist_len > 0:
            ghist_size_log = int(np.ceil(np.log2(self.max_hist_len)))
        else:
            ghist_size_log = 1
        self.ghist_bufsize = 2**ghist_size_log
        self.ghist = np.zeros(self.ghist_bufsize ,dtype = np.uint8)

        #self.phist = np.uint32(0)
        self.metadata[0]['ghist_size_log'] = ghist_size_log
        self.metadata[0]['phist_len'] = spec['global_config']['phist_len']
        self.metadata[0]['u_tick_log'] = spec['global_config']['u_duration_log']

        self.logger.info(f'hist_len_arr:\n    {self.hist_len_arr}')
        self.logger.info(f'comp_hist_idx:\n    {self.comp_hist_idx_arr}')
        self.logger.info(f'comp_hist_tag0:\n   {self.comp_hist_tag0_arr}')
        self.logger.info(f'comp_hist_tag1:\n   {self.comp_hist_tag1_arr}')
        self.logger.info(f'tagged_offsets:\n    {self.tagged_offsets}')
        self.logger.info(f'base_size:\n    {len(self.base_entries)}')
        self.logger.info(f'tagged_size:\n    {len(self.tagged_entries)}')
        self.logger.info(f'tag_widths:\n    {self.tagged_tag_widths}')

        # Size calculation:
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

        self.storage_report['tot_size_b'] = total_storage #((sum(self.storage_report.values())) / 1024)# / 8192
        
        self.logger.info("storage_info:")
        for k,v in self.storage_report.items():
            self.logger.info(f"    {k}: {v}")

        return

