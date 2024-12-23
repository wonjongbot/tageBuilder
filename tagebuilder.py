#import random
import numpy as np
import settings
from typing import List, Optional, Dict, Tuple
from numpy.typing import NDArray
import logging

# Basic circular buffer for global history register (GHR)
class CircularBuffer:
    def __init__(self, bufsize_log, dtype):
        self.dtype = dtype
        self.bufsize = 2**bufsize_log
        self.head = self.bufsize - 1    # head is next available spot
        self.buf = np.zeros(self.bufsize, dtype=self.dtype)

    def getBuf(self, length:int) -> NDArray:
        """
        return a np array with 0th element being head and <length - 1>th
        entry being last element of the array
        """
        idx = (self.head + 1 + np.arange(length)) % self.bufsize
        return self.buf[idx]

    def updateBuf(self, val:int):
        """
        update buffer with new value
        """
        self.buf[self.head] = self.dtype(val)
        self.head = (self.head - 1) % self.bufsize
    
    def getVal(self, idx) -> Optional[int]:
        return self.buf[(1 + self.head + idx) % self.bufsize]

# Compressed history class for tagged components with desired history length
class CompressedHistory:
    def __init__(self, orig_len:Optional[int], comp_len:Optional[int]):
        self.comp = 0
        self.orig_len = int(orig_len)
        self.comp_len = int(comp_len)
        self.offset = self.orig_len % self.comp_len
        self.mask = (1 << self.comp_len) - 1

    def update2(self, buf:CircularBuffer, youngestIdx:int, oldestIdx:int):
        """
        update the compressed history -- history is folded when it reaches
        its max length 
        """
        youngest = buf.buf[(1 + buf.head + youngestIdx) & (buf.bufsize - 1)]
        oldest = buf.buf[(1 + buf.head + oldestIdx) & (buf.bufsize - 1)]
        
        self.comp = (self.comp << 1) | int(youngest)
        self.comp ^= int(oldest) << self.offset
        self.comp ^= (self.comp >> self.comp_len)
        self.comp &= self.mask

# TAGE predictor class with customizable parameters
class TAGEPredictor:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}")
        self.logger.info("Initializing TAGEPredictor")

        self.rng = np.random.default_rng()
        self.rand_array = self.rng.integers(low=0, high=3, size=10000)
        self.rand_index = 0
        self.sizelog = ''

        self.tables = []
        self.word_align = 18
        self.branch_pc = 0
        self.base_idx_width = 14
        self.next_state_bimodal =   {   (0, 0, True):  (0, 1),
                                        (0, 0, False): (0, 0),
                                        (0, 1, True):  (1, 0),
                                        (0, 1, False): (0, 1),
                                        (1, 0, True):  (1, 1),
                                        (1, 0, False): (0, 1),
                                        (1, 1, True):  (1, 1),
                                        (1, 1, False): (1, 0) }
        self.id_gen = 1
        self.comp_hist_idx = []
        self.comp_hist_tag = [[],[]]
        self.phist = np.uint16(0)
        self.ghist_entries = 10
        self.ghist = CircularBuffer(self.ghist_entries, np.uint8)       
        self.ghist_len = 0
        self.phist_len = 16
        self.ghist_ptr = -1

        self.rand = 0

        self.bim_pred = False

        self.tagged_predictors = []
        self.id2name = [None]*32
        self.name2id = {}
        self.use_alt_on_new_alloc = 0

        self.hitpred_id = 0
        self.hitpred_ctr = -1
        self.hitpred_taken = None

        self.altpred_id = 0
        self.altpred_ctr = -1
        self.altpred_taken = None

        self.tage_pred = None

        self.num_tagged = 0

        self.u_tick = 0
        self.u_tick_log = 19

        self.base_pid = 0

    def init_tables(self, cfg:Dict):
        """
        declare and initialize predictor tables; calculate estimated memory footprint 
        """
        log = ''
        total_size = 0 # in Kibibits

        max_id = max(value['id'] for value in cfg.values())
        self.tables = [None] * (max_id + 1)
        self.comp_hist_idx = [None] * (max_id + 1)
        self.comp_hist_tag[0] = [None] * (max_id + 1)
        self.comp_hist_tag[1] = [None] * (max_id + 1)

        for key, value in cfg.items():
            pid = value['id']
            self.name2id[key] = pid
            if key == 'base':
                self.base_pid = pid
                assert not value['isTaggedComp']
                self.tables[pid] = {}
                self.tables[pid]['pred'] = np.zeros(2**value['ent_pred'], dtype=np.uint8)
                self.tables[pid]['hyst'] = np.ones(2**value['ent_hyst'], dtype=np.uint8)
                self.logger.info(f"Initialized pred array of size {len(self.tables[pid]['pred'])} and hyst array of size {len(self.tables[pid]['hyst'])}")
                self.tables[pid]['predWidth'] = 1 # considers pred table and hyst table as one
                self.tables[pid]['hystWidth'] = 1
                self.tables[pid]['ent_pred'] = value['ent_pred']
                self.tables[pid]['ent_hyst'] = value['ent_hyst']
                self.tables[pid]['hist_len'] = value['hist_len']
                self.tables[pid]['id'] = value['id']
                self.id2name[value['id']] = key

                table_size = (2**value['ent_pred'] + 2**value['ent_hyst']) / 2**10
                total_size += table_size
                log += f'{key} =\t{table_size}Kb\n'
            elif value['isTaggedComp']:
                self.num_tagged += 1

                pred_bits = 3
                tag_bits = value['tag_width']
                u_bits = 2
                total_bits = pred_bits+tag_bits+u_bits
                num_ent = 2**value['ent_pred']

                self.tables[pid] = {}

                # Create Tags based on the actual branches so I can access dict to find 
                #   useful bit and more

                # entries structure:
                #        [{
                #           tag: np.uint16(tag)
                #           pred:   np.unit8(pred)
                #           u:      np.unit8(u)
                #        }, ...]
                
                # initialized as weakly not taken
                customdtype = np.dtype([('tag', np.uint16), ('pred', np.uint8), ('u', np.uint8)])
                self.tables[pid]['entries'] = np.zeros(num_ent, dtype=customdtype)
                self.tables[pid]['entries']['pred'][:] = 3

                self.tables[pid]['tagWidth'] = tag_bits
                self.tables[pid]['predWidth'] = 3
                self.tables[pid]['isTaggedComp'] = True
                self.tables[pid]['ent_pred'] = value['ent_pred']
                self.tables[pid]['hist_len'] = value['hist_len']
                self.tables[pid]['id'] = value['id']
                self.tagged_predictors.append(key)
                self.id2name[value['id']] = key

                self.ghist_len = self.tables[pid]['hist_len'] if self.tables[pid]['hist_len'] > self.ghist_len else self.ghist_len

                self.comp_hist_idx[pid] = CompressedHistory(value['hist_len'], value['ent_pred'])
                self.comp_hist_tag[0][pid] = CompressedHistory(value['hist_len'], tag_bits)
                self.comp_hist_tag[1][pid] = CompressedHistory(value['hist_len'], tag_bits - 1)

                assert (total_bits == self.tables[pid]['predWidth'] + self.tables[pid]['tagWidth'] + 2)

                table_size = ((self.tables[pid]['predWidth'] + self.tables[pid]['tagWidth'] + 2) * (num_ent)) / 2**10
                total_size += table_size
                id = value['id']
                log += f'id: {id} :: {key} =\t{table_size}Kb\n'
        
        self.logger.info(self.name2id)

        self.tage_idx = [0] * (self.num_tagged + 1)
        self.tage_tag = [0] * (self.num_tagged + 1)
        
        self.sizelog = f'\n{log}\nTotal Size: {total_size}Kb\nLongest history length: {self.ghist_len}\nnum tagged comp: {self.num_tagged}\n'
        self.logger.info(self.sizelog)
    
    def mix_path_history(self, predictor_name:str, phist_size:int, phist:np.uint16)->int:
        """
        path history (history of a single bit from instruction address) hash generation
        for tagged entry access
        """
        phist_c = int(phist & ((1 << phist_size) - 1))
        pid = self.name2id[predictor_name]
        size = int(self.tables[pid]['ent_pred'])
        bank = int(pid)
        mask = (1 << size) - 1

        A = phist_c
        A1 = A & mask
        A2 = A >> size
        A2 = ((A2 << bank) & (mask)) + (A2 >> abs(size - bank))
        A = A1 ^ A2
        A = ((A << bank) & (mask)) + (A >> abs(size - bank))
        
        #self.logger.debug(f'F - {A}')
        return A

    def mix_path_history_simplified(self, predictor_name, phist_size:int, phist:np.uint16)->int:
        """
        a simplified version of mix_path_history()
        """
        # Mask the path history to the given size
        phist_c = phist & ((1 << phist_size) - 1)
        
        # Perform a single XOR with a predefined constant instead of multiple shifts
        mixed = phist_c ^ (phist_c >> 2)  # Simplified mixing logic
        
        # Mask the result to limit its size
        return int(mixed & ((1 << phist_size) - 1))
    
    def get_taggedComp_tag(self, predictor_name:str, bpc_hashed:int)->int:
        """
        calculate entry tag based on compressed history and branch address
        """
        pid = self.name2id[predictor_name]
        tag = bpc_hashed ^ self.comp_hist_tag[0][pid].comp ^ (self.comp_hist_tag[1][pid].comp << 1)
        pid = self.name2id[predictor_name]
        return (tag & ((1 << self.tables[pid]['tagWidth']) - 1))

    def get_taggedComp_idx(self, predictor_name:str, bpc_hashed:int)->int:
        """
        index for tagged component based on predictor id, branch address, and compressed history
        """
        pid = self.name2id[predictor_name]
        table = self.tables[pid]
        hist_len = min(16, table['hist_len'])
        
        foo = (bpc_hashed >> (abs(table['ent_pred'] - pid) + 1))
        idx = bpc_hashed ^ foo ^ self.comp_hist_idx[pid].comp ^ self.mix_path_history(predictor_name, hist_len, self.phist)

        return (idx & ((1 << table['ent_pred'])-1))

    def predict_bimodal(self, branch_pc:int)->bool:
        """
        access prediction from the base bimodal predictor
        """
        idx_bimodal = (branch_pc >> self.word_align) & ((1<<self.base_idx_width) - 1)
        pred_b = self.tables[self.base_pid]['pred'][idx_bimodal]

        self.bim_pred = bool(pred_b)

        return bool(pred_b)

    def predict_tagged(self, predictor_name:str, branch_pc:int)->Tuple[bool, Optional[np.uint8]]:
        """
        predict branch outcome for <predictor_name>
        """
        isHit = False
        pred = None

        pid = self.name2id[predictor_name]
        idx = self.tage_idx[self.tables[pid]['id']]
        tag = self.tage_tag[self.tables[pid]['id']]

        value = self.tables[pid]['entries'][idx]

        #print(value, tag)
        if value["tag"] == np.uint16(tag):
            isHit = True
            pred = value["pred"]
        
        #self.logger.debug(f'value at {predictor_name}, {value}')
        
        return(isHit, pred)

    def make_prediction(self)->bool:
        """
        generate final prediction among results from base and tagged predictors
        """
        # BIMODAL PREDICTOR prediction
        self.predict_bimodal(self.branch_pc)
        #self.logger.debug(f'------------------------')
        #self.logger.debug(f'bpc: {(hex(self.branch_pc))}')

        self.hitpred_id = 0
        self.hitpred_ctr = None
        self.hitpred_taken = None

        self.altpred_id = 0
        self.altpred_ctr = None
        self.altpred_taken = None

        # Fold branch PC to calculate tag and index
        bpc_hashed = (self.branch_pc ^ (self.branch_pc >> 16))

        for predictor_name in self.tagged_predictors:
            # update tag and idx for each predictor for current branch for update
            pid = self.name2id[predictor_name]
            self.tage_idx[pid] = self.get_taggedComp_idx(predictor_name, bpc_hashed)
            self.tage_tag[pid] = self.get_taggedComp_tag(predictor_name, bpc_hashed)
        
        #self.logger.debug(f'idx info { self.tage_idx}')
        #self.logger.debug(f'tag info { self.tage_tag}')

        # Look for main predictor hit
        for id in range(self.num_tagged, 0, -1):
            entry = self.tables[id]['entries'][self.tage_idx[id]]
            if entry['tag'] == self.tage_tag[id]:
                #self.logger.debug(f"TAG MATCH FOUND ON predictor {id} :: idx {self.tage_idx[id]} : {self.tage_tag[id]}")
                self.hitpred_id = id
                self.hitpred_ctr = entry['pred']
                self.hitpred_taken = True if self.hitpred_ctr > 3 else False
                break
        
        # Look for alternate predictor hit
        for id in range(self.hitpred_id - 1, 0, -1):
            entry = self.tables[id]['entries'][self.tage_idx[id]]
            if entry['tag'] == self.tage_tag[id]:
                #self.logger.debug(f"ALT TAG MATCH FOUND ON predictor {id} :: idx {self.tage_idx[id]} : {self.tage_tag[id]}")
                # only use alternate predictor value if use_alt ctr is positive and prediction is confident enough
                # NOTE: Seznec's code has opposite logic for USE_ALT_ON_NA -- why?
                if (self.use_alt_on_new_alloc >= 0) and (entry['pred'] not in (3, 4)):
                #if (self.use_alt_on_new_alloc < 0) or (entry['pred'] not in (3, 4)):
                    #self.logger.debug("Not using alt :: entry is not newalloc")
                    self.altpred_id = id
                    self.altpred_ctr = entry['pred']
                    self.altpred_taken = True if self.altpred_ctr > 3 else False
                    break
        
        if self.hitpred_id > 0:
            # if no altpred is found, altpred is bimodal predictor
            if self.altpred_id <= 0:
                #self.logger.debug("ALTPRED IS BIMODAL")
                self.altpred_taken = self.bim_pred

            # Use main hit predictor if use_alt counter is negative or
            #    3-bit prediction ctr is weak (i.e. potential new allocation) 
            if (self.use_alt_on_new_alloc < 0) or (self.hitpred_ctr not in (3,4)):
                #self.logger.debug("TAGEPRED is MAIN HIT PREDICTOR")
                self.tage_pred = True if self.hitpred_ctr > 3 else False
            else:
                #self.logger.debug("TAGEPRED IS ALTPRED")
                self.tage_pred = self.altpred_taken
        else:
            #self.logger.debug("NO HIT DETECTED. TAGE PRED == ALTPRED == BIMODAL")
            self.altpred_taken = self.bim_pred
            self.tage_pred = self.altpred_taken

        return self.tage_pred

    def update_bimodal(self, isTaken:bool):
        """
        update bimodal predictor based on branch outcome
        """
        idx_bimodal = (self.branch_pc >> self.word_align) & ((1<<self.base_idx_width) - 1)
        pred_b = self.tables[self.base_pid]['pred'][idx_bimodal]
        #hyst_b = self.tables['base']['hyst'][idx_bimodal >> self.tables['base']['ent_hyst']]
        hyst_b = self.tables[self.base_pid]['hyst'][idx_bimodal >> 2]

        (pred_b, hyst_b) = self.next_state_bimodal[(pred_b, hyst_b, isTaken)]

        self.tables[self.base_pid]['pred'][idx_bimodal] = pred_b
        #self.tables['base']['hyst'][idx_bimodal >> self.tables['base']['ent_hyst']] = hyst_b
        self.tables[self.base_pid]['hyst'][idx_bimodal >> 2] = hyst_b

    def update_ghist(self, isTaken:bool):
        """
        update GHR based on branch outcome
        """
        self.ghist_ptr = self.ghist.head
        self.ghist.updateBuf(np.uint8(isTaken))

        #self.logger.debug(f'ptr: {self.ghist_ptr}')
        #self.logger.debug(f'GHIST: {self.ghist.getBuf(16)}')
    
    def update_phist(self, branch_pc:int):
        """
        update path history register based on branch instruction address
        """
        phist = (int(self.phist) << 1) | ((branch_pc ^ (branch_pc >> 16)) & 1)
        
        #phist = (int(self.phist) << 1) | ((branch_pc) & 1)
        self.phist = np.uint16(phist & ((1<<self.phist_len) - 1))
        
        # if self.logger.isEnabledFor(logging.DEBUG):
            #self.logger.debug(hex(branch_pc))
            #tmp = int(self.phist)
            #binstr = ''
            #for i in range(self.phist_len):
                #binstr = str(tmp&1) + binstr
                #tmp >>= 1
            #self.logger.debug(f'phist: 0b{binstr}')
    
    def update_tage_ctr(self, isTaken:bool, id:int):
        """
        update 3-bit counter (representing weak-strong prediction outcome) of tagged predictors' entries
        """
        ctr = self.tables[id]['entries'][self.tage_idx[id]]['pred']
        entry = self.tables[id]['entries'][self.tage_idx[id]]

        if isTaken:
           if ctr < 7:
               entry['pred'] += 1
        else:
            if ctr > 0:
                entry['pred'] -= 1
        # if self.logger.isEnabledFor(logging.DEBUG):
            #foo = entry['pred']
            #self.logger.debug(f'ctr update: {ctr} ->  {foo}')


    def update_tagged(self, isTaken:bool):
        """
        update tagged predictors based on branch outcome
        """
        # new entry is allocated if prediction is wrong and there is a predictor with longer GH
        isNewAlloc = ((self.tage_pred != isTaken) and (self.hitpred_id < self.num_tagged))

        # if predictor used for prediction is tagged (not base/bimodal)
        if (self.hitpred_id > 0):
            assert(self.hitpred_ctr >= 0) # make sure predicted value is valid (member var initialized as -1)

            hitpred_taken = self.hitpred_taken 
            
            foo  = True if self.hitpred_ctr > 3 else False
            assert (foo == self.hitpred_taken) # make sure member variables are consistent -- counter value(int) vs prediction value(bool)
            
            altpred_taken = self.altpred_taken

            # prediction ctr of 3 or 4 means its either weakly correlated or is a new prediction
            pseudoNewAlloc = True if (self.hitpred_ctr in (3,4)) else False 

            # When new allocation is detected
            if(pseudoNewAlloc):
                # train current hit predictor; not need to create a new entry in higher rank predictor
                if(hitpred_taken == isTaken):
                    isNewAlloc = False
                if(hitpred_taken != altpred_taken):
                    # if alternative predictor got it right but hitpred didn't,
                    #   increase bias on using alternative predictor
                    if(altpred_taken == isTaken):
                        if (self.use_alt_on_new_alloc < 7):
                            self.use_alt_on_new_alloc += 1
                    elif(self.use_alt_on_new_alloc > -8):
                        self.use_alt_on_new_alloc -= 1
                if(self.use_alt_on_new_alloc >= 0):
                    self.tage_pred = hitpred_taken
        
        # allocate new entry
        if (isNewAlloc):
            min_u = 1

            # use numpy arr to vectorize finding min value
            u_values = np.array([self.tables[i]['entries'][self.tage_idx[i]]['u'] for i in range(self.hitpred_id + 1, self.num_tagged + 1)])
            min_u = np.min(u_values)
            
            # find entry to allocate (select among 3 tables with longer history)
            # Randomly select a bank among the next longer history tables
            diff = int(self.num_tagged - self.hitpred_id)
            num_banks_to_consider = min(3, diff)
            random_offset = self.rand % num_banks_to_consider
            newentryId = self.hitpred_id + 1 + random_offset 
            
            # allocate new entry if none available
            if min_u > 0:
                self.tables[newentryId]['entries'][self.tage_idx[newentryId]]['u'] = 0
            
            for i in range(newentryId, self.num_tagged + 1):
                entry = self.tables[i]['entries'][self.tage_idx[i]]
                if (entry['u'] == 0):
                    entry['tag'] = np.uint16(self.tage_tag[i])
                    entry['pred'] = np.uint8(4) if (isTaken) else np.uint8(3)
                    entry['u'] = np.uint8(0)
                    # if self.logger.isEnabledFor(logging.DEBUG):
                        #self.logger.debug(f'settings.DEBUG: {i} {newentryId}')
                        #foo = entry['tag']
                        #self.logger.debug(f'ENTRY CREATED at ID {i}, TAG {foo}')
                    break
        ## ALLOCATE DONE
        
        # RESET useful bit when tick is saturated 
        self.u_tick += 1
        if ((self.u_tick & ((1 << self.u_tick_log) - 1)) == 0):
            #self.logger.debug('RESETTING UBIT')
            for i in range(1, self.num_tagged + 1):
                for j in range(0, (1 << self.tables[i]['ent_pred'])):
                    self.tables[i]['entries'][j]['u'] >> 1
        # RESET ubit done

        # UPDATE COUNTERS
        if self.hitpred_id > 0:
            entry_update = self.tables[self.hitpred_id]['entries'][self.tage_idx[self.hitpred_id]]
            ## update the hit counter
            self.update_tage_ctr(isTaken, self.hitpred_id)
            # if hit counter is not "useful" anymore, also update alterate predictor
            if entry_update['u'] == 0:
                if self.altpred_id > 0:
                    self.update_tage_ctr(isTaken, self.altpred_id)
                elif self.altpred_id == 0:
                    # update bimodal
                    self.update_bimodal(isTaken)
        else:
            self.update_bimodal(isTaken)
        

        # UPDATE u counter
        #   Only update useful bit if main prediction is different from alternate prediction
        if self.tage_pred != self.altpred_taken:
            if self.tage_pred == isTaken:
                if entry_update['u'] < 3:
                    entry_update['u'] += 1
                else:
                    if self.use_alt_on_new_alloc < 0:
                        if (entry_update['u'] > 0):
                            entry_update['u'] -= 1


        # update ghist and phist
        self.update_ghist(isTaken)
        self.update_phist(self.branch_pc)
        
        # update compressed histories
        for i in range(1, self.num_tagged + 1):
            self.comp_hist_idx[i].update2(self.ghist, 0, self.tables[i]['hist_len'])
            self.comp_hist_tag[0][i].update2(self.ghist, 0, self.tables[i]['hist_len'])
            self.comp_hist_tag[1][i].update2(self.ghist, 0, self.tables[i]['hist_len'])


        return 
    
    def train_predictor(self, isTaken):
        
        # USE LSFR for RTL implementation
        # use cached random value
        self.rand = self.rand_array[self.rand_index]
        self.rand_index = (self.rand_index + 1) % len(self.rand_array)

        #self.logger.debug(f'RAND: {self.rand}\nISTAKEN: {isTaken}')
        self.update_tagged(isTaken)

        return 0