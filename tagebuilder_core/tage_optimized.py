import yaml
import numba
import numpy as np
from numba import prange
from numba import njit

### SECTINO: GLOBALS
metadtype = np.dtype([
    ('base_idx_mask', np.uint32),
    ('base_pred_hyst_diff_log', np.uint32)
    ])
### GLOBAL DONE

### SECTION: CORE LOGIC
@njit(inline='always')
def mix_path_history(
    len,
    phist      
    ):
    phist_masked = phist & ((1 << len) - 1)

    mixed = phist_masked ^ (phist >> 2)
                            
    return np.uint32(mixed & ((1 << len) - 1))

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
    return (bpc ^ foo ^ t_comp_hist_idx ^ mix_path_history(t_hist_len, phist)) & mask

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
    phist,
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
            return np.array([np.int8(e['pred_ctr'] >> 2), np.int8(1), e['pred_ctr']], dtype=np.int8)
        else:
            return np.array([np.int8(0), np.int8(0), np.int8(-1)], dtype=np.int8)

@njit(inline='always')
def update_predictor(
    branch_pc,
    branch_taken,
    predictor_id,
    base_entries,
    tagged_entries,
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
    phist,
    use_alt_on_new_alloc,
    metadata
    ):
    #predictor_id = 0
    results = np.zeros(len(br_infos), dtype=np.uint8)
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
                phist,
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

        # CHANGE RETURN VALUE SO IT GIVES COUNTER ITSELF??
        if main_pred_id > 0:
            if (use_alt_on_new_alloc < 0) or (predictions[int(main_pred_id), 2] not in (3,4)):
                final_pred = main_pred
            else:
                final_pred = alt_pred
        else:
            final_pred = alt_pred

        #print(pred, b['taken'])
        results[i] = np.uint8(final_pred == b['taken'])
        #print(results[i])
        update_predictor(
            b['addr'],
            b['taken'],
            predictor_id,
            base_entries,
            tagged_entries,
            metadata
        )
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
@njit
def comp_hist_init(comp_hist, orig_len:np.uint32, comp_len:np.uint32):
    comp_hist['comp'] = 0
    comp_hist['orig_len'] = orig_len
    comp_hist['comp_len'] = comp_len
    comp_hist['offset'] = orig_len % comp_len
    print(comp_len)
    comp_hist['mask'] = (1 << comp_len) -1

def comp_hist_update(comp_hist):
    pass
### COMPRESSED HISTORY DONE


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
    def __init__(self, spec):
        self.storage_report = {}
        self.phist_len = spec['global_config']['phist_len']
        self.u_duration_log = spec['global_config']['u_duration_log']

        self.num_tables = len(spec['tables'])
        self.num_tot_tagged_entries = 0
        self.id2name = []
        self.max_hist_len = np.uint32(0)

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

        self.use_alt_on_new_alloc = np.int8(0)
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
        self.ghist_bufsize = 2**10
        self.ghist = np.zeros(self.ghist_bufsize ,dtype = np.uint8)
        self.phist = np.uint32(0)

        print(f'hist_len_arr:\n    {self.hist_len_arr}')
        print(f'comp_hist_idx:\n    {self.comp_hist_idx_arr}')
        print(f'comp_hist_tag0:\n   {self.comp_hist_tag0_arr}')
        print(f'comp_hist_tag1:\n   {self.comp_hist_tag1_arr}')
        print(f'tagged_offsets:\n    {self.tagged_offsets}')
        print(f'base_size:\n    {len(self.base_entries)}')
        print(f'tagged_size:\n    {len(self.tagged_entries)}')
        print(f'tag_widths:\n    {self.tagged_tag_widths}')

        # Size calculation:
        self.storage_report['ghist_size_b'] = self.max_hist_len
        self.storage_report['phist_size_b'] = self.phist_len
        self.storage_report['use_alt_on_new_alloc'] = 4
        self.storage_report['base_size_pred_b'] = 2**self.base_num_pred_entries_log
        self.storage_report['base_size_hyst_b'] = 2**self.base_num_hyst_entries_log
        for id in range(1, self.num_tables):
            if id < self.num_tables - 1:
                self.storage_report[f'tagged_{id}_size_b'] = (3 + 2 + self.tagged_tag_widths[id])*(self.tagged_offsets[id + 1] - self.tagged_offsets[id])
            else:
                self.storage_report[f'tagged_{id}_size_b'] = (3 + 2 + self.tagged_tag_widths[id])*(len(self.tagged_entries) - self.tagged_offsets[id])
        self.storage_report['tot_size_Kb'] = (sum(self.storage_report.values())) / 1024# / 8192
        
        print("storage_info:")
        for k,v in self.storage_report.items():
            print(f"    {k}: {v}")

        return



if __name__ == "__main__":
    spec = None
    spec_src = "/home/wonjongbot/tageBuilder/configs/tage_l.yaml"
    with open(spec_src, "r") as f:
        spec = yaml.safe_load(f)
    
    for k,v in spec.items():
        print(f'{k}')
        if k == 'global_config':
            for k1, v1 in v.items():
                print(f'    {k1}: {v1}')
        if k == 'tables':
            for i, t in enumerate(v):
                print(f'    id: {i}')
                for k1, v1 in t.items():
                    print(f'        {k1}: {v1}')

    predictor = TAGEPredictor(spec)

    idx = 10
    pred = get_prediction(idx, 0, predictor.base_entries, None, predictor.metadata[0])
    print(pred)
    print(get_entry(predictor, 0, idx))
    update_predictor(idx, True, 0, predictor.base_entries, None, predictor.metadata[0])
    print(get_entry(predictor, 0, idx))
    
    print(get_prediction(idx, 0, predictor.base_entries, None, predictor.metadata[0]))

