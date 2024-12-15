#import random
import numpy as np
import settings

class CircularBuffer:
    def __init__(self, bufsize_log, dtype):
        self.dtype = dtype
        self.bufsize = 2**bufsize_log
        # head is next available spot
        self.head = self.bufsize - 1
        self.buf = np.zeros(self.bufsize, dtype=self.dtype)

    def getBuf(self, length):
        idx = (self.head + 1 + np.arange(length)) % self.bufsize
        return self.buf[idx]

    def updateBuf(self, val):
        uintval = self.dtype(val)
        self.buf[self.head] = uintval
        #print(f'ADDED {val} to idx {self.head}')
        self.head = (self.head - 1) % self.bufsize
    
    def getVal(self, idx):
        return self.buf[(1 + self.head + idx) % self.bufsize]

class CompressedHistory:
    def __init__(self, orig_len, comp_len):
        self.comp = 0
        self.orig_len = int(orig_len)
        self.comp_len = int(comp_len)
        self.offset = self.orig_len % self.comp_len
        self.mask = (1 << self.comp_len) - 1
    def update(self, hist, idx):

        self.comp = (self.comp << 1) | int(hist[0])
        #self.comp ^= int(hist[idx + self.orig_len]) << self.offset
        self.comp ^= int(hist[self.orig_len]) << self.offset
        self.comp ^= (self.comp >> self.comp_len)
        self.comp &= self.mask

    def update2(self, buf, youngestIdx, oldestIdx):
        youngest = buf.buf[(1 + buf.head + youngestIdx) & (buf.bufsize - 1)]
        oldest = buf.buf[(1 + buf.head + oldestIdx) & (buf.bufsize - 1)]
        
        self.comp = (self.comp << 1) | int(youngest)
        #self.comp ^= int(hist[idx + self.orig_len]) << self.offset
        self.comp ^= int(oldest) << self.offset
        self.comp ^= (self.comp >> self.comp_len)
        self.comp &= self.mask

class TAGEPredictor:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.rand_array = self.rng.integers(low=0, high=3, size=10000)
        self.rand_index = 0
        self.sizelog = ''

        self.tables = {}
        self.word_align = 18 # OPTIMIZATION: use middle of the PC to decrease aliasing
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
        self.comp_hist_idx = {}
        self.comp_hist_tag = [{},{}]
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

    def init_tables(self, cfg):
        log = ''
        total_size = 0 # in Kibibits
        for key, value in cfg.items():
            #print(key, value)
            if key == 'base':
                assert not value['isTaggedComp']
                self.tables[key] = {}
                self.tables[key]['pred'] = np.zeros(2**value['ent_pred'], dtype=np.uint8)
                self.tables[key]['hyst'] = np.ones(2**value['ent_hyst'], dtype=np.uint8)
                print(f"Initialized pred array of size {len(self.tables[key]['pred'])} and hyst array of size {len(self.tables[key]['hyst'])}")
                self.tables[key]['predWidth'] = 1 # considers pred table and hyst table as one
                self.tables[key]['hystWidth'] = 1
                self.tables[key]['ent_pred'] = value['ent_pred']
                self.tables[key]['ent_hyst'] = value['ent_hyst']
                self.tables[key]['hist_len'] = value['hist_len']
                self.tables[key]['id'] = value['id']
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

                self.tables[key] = {}

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
                self.tables[key]['entries'] = np.zeros(num_ent, dtype=customdtype)
                self.tables[key]['entries']['pred'][:] = 3

                # NOTE PREVIOUS IMPLEMENTATION OF ENTRIES
                #self.tables[key]['entries'] = [{'tag': np.uint16(0), 'pred': np.uint8(3), 'u':np.uint8(0)} for i in range(2**value['ent_pred'])]


                #self.tables[key]['pred'] = np.zeros(num_ent, dtype=np.uint8)
                #self.tables[key]['tag'] = np.zeros(num_ent, dtype=np.uint32)
                #self.tables[key]['u'] = np.zeros(num_ent, dtype=np.uint8)
                self.tables[key]['tagWidth'] = tag_bits
                self.tables[key]['predWidth'] = 3
                self.tables[key]['isTaggedComp'] = True
                self.tables[key]['ent_pred'] = value['ent_pred']
                self.tables[key]['hist_len'] = value['hist_len']
                self.tables[key]['id'] = value['id']
                self.tagged_predictors.append(key)
                self.id2name[value['id']] = key
                #self.id_gen += 1
                self.ghist_len = self.tables[key]['hist_len'] if self.tables[key]['hist_len'] > self.ghist_len else self.ghist_len

                self.comp_hist_idx[key] = CompressedHistory(value['hist_len'], value['ent_pred'])
                self.comp_hist_tag[0][key] = CompressedHistory(value['hist_len'], tag_bits)
                self.comp_hist_tag[1][key] = CompressedHistory(value['hist_len'], tag_bits - 1)

                assert (total_bits == self.tables[key]['predWidth'] + self.tables[key]['tagWidth'] + 2)

                table_size = ((self.tables[key]['predWidth'] + self.tables[key]['tagWidth'] + 2) * (num_ent)) / 2**10
                total_size += table_size
                id = value['id']
                log += f'id: {id} :: {key} =\t{table_size}Kb\n'
        
        self.tage_idx = [0] * (self.num_tagged + 1) # np.zeros(self.num_tagged+1) #[0] * (self.num_tagged + 1)
        self.tage_tag = [0] * (self.num_tagged + 1) # np.zeros(self.num_tagged+1) #[0] * (self.num_tagged + 1)
        
        self.sizelog = f'{log}\nTotal Size: {total_size}Kb\nLongest history length: {self.ghist_len}\nnum tagged comp: {self.num_tagged}\n'
        print(self.sizelog)
    
    def mix_path_history(self, predictor_name, phist_size, phist):
        phist_c = int(phist & ((1 << phist_size) - 1))
        size = int(self.tables[predictor_name]['ent_pred'])
        bank = int(self.tables[predictor_name]['id'])
        A = phist_c
        A1 = A & ((1 << size) - 1)
        A2 = A >> size
        A2 = ((A2 << bank) & ((1 << size) - 1)) + (A2 >> abs(size - bank))
        A = A1 ^ A2
        A = ((A << bank) & ((1 << size) - 1)) + (A >> abs(size - bank))
        
        if settings.DEBUG == 1:
            print(f'F - {A}')
        return A
    
    def get_taggedComp_tag(self, predictor_name, bpc_hashed):
        #bpc_hashed = (branch_pc ^ (branch_pc >> 16))
        tag = bpc_hashed ^ self.comp_hist_tag[0][predictor_name].comp ^ (self.comp_hist_tag[1][predictor_name].comp << 1)
        return (tag & ((1 << self.tables[predictor_name]['tagWidth']) - 1))

    def get_taggedComp_idx(self, predictor_name, bpc_hashed):
        table = self.tables[predictor_name]
        #bpc_hashed = (branch_pc ^ (branch_pc >> 16))
        hist_len = min(16, table['hist_len']) #16 if (self.tables[predictor_name]['hist_len'] > 16) else self.tables[predictor_name]['hist_len']
        
        foo = (bpc_hashed >> (abs(table['ent_pred'] - table['id']) + 1))
        idx = bpc_hashed ^ foo ^ self.comp_hist_idx[predictor_name].comp ^ self.mix_path_history(predictor_name, hist_len, self.phist)

        return (idx & ((1 << table['ent_pred'])-1))

    def predict_bimodal(self, branch_pc):
        idx_bimodal = (branch_pc >> self.word_align) & ((1<<self.base_idx_width) - 1)
        pred_b = self.tables['base']['pred'][idx_bimodal]

        self.bim_pred = bool(pred_b)

        return bool(pred_b)

    def predict_tagged(self, predictor_name, branch_pc):
        isHit = False
        pred = None

        # update tag and idx for each predictor for current branch for update
        #for predidctors in self.tagged_predictors:
        #    self.tage_idx[self.tables[predidctors]['id']] = self.get_taggedComp_idx(predidctors, branch_pc)
        #    self.tage_tag[self.tables[predidctors]['id']] = self.get_taggedComp_tag(predidctors, branch_pc)
        #if settings.DEBUG == 1:
        #    print(f'idx info { self.tage_idx}')
        #    print(f'tag info { self.tage_tag}')


        idx = self.tage_idx[self.tables[predictor_name]['id']]
        tag = self.tage_tag[self.tables[predictor_name]['id']]

        value = self.tables[predictor_name]['entries'][idx]

        #print(value, tag)
        if value["tag"] == np.uint16(tag):
            isHit = True
            pred = value["pred"]
        
        if settings.DEBUG == 1:
            print(f'value at {predictor_name}, {value}')
        
        # return a bool and int. one to represent hit for partial tag and other for prediction ctr
        return(isHit, pred)

    def make_prediction(self):
        # BIMODAL PREDICTOR
        #pred_bimodal = 
        self.predict_bimodal(self.branch_pc)
        #return pred_bimodal
        if settings.DEBUG == 1:
            print(f'------------------------\nbpc: {(hex(self.branch_pc))}')

        self.hitpred_id = 0
        self.hitpred_ctr = None
        self.hitpred_taken = None

        self.altpred_id = 0
        self.altpred_ctr = None
        self.altpred_taken = None

        #self.use_alt_on_new_alloc

        # Fold branch PC to calculate tag and index
        bpc_hashed = (self.branch_pc ^ (self.branch_pc >> 16))

        for predictor_name in self.tagged_predictors:
            # update tag and idx for each predictor for current branch for update
            self.tage_idx[self.tables[predictor_name]['id']] = self.get_taggedComp_idx(predictor_name, bpc_hashed)
            self.tage_tag[self.tables[predictor_name]['id']] = self.get_taggedComp_tag(predictor_name, bpc_hashed)
        
        if settings.DEBUG == 1:
            print(f'idx info { self.tage_idx}')
            print(f'tag info { self.tage_tag}')

        # Look for main hit
        for id in range(self.num_tagged, 0, -1):
            #print(id)
            #print(self.tables[self.id2name[id]]['entries'][self.tage_idx[id]]['tag'])
            if self.tables[self.id2name[id]]['entries'][self.tage_idx[id]]['tag'] == self.tage_tag[id]:
                if settings.DEBUG == 1:
                    print(f"TAG MATCH FOUND ON predictor {id} :: idx {self.tage_idx[id]} : {self.tage_tag[id]}")
                self.hitpred_id = id
                self.hitpred_ctr = self.tables[self.id2name[id]]['entries'][self.tage_idx[id]]['pred']
                self.hitpred_taken = True if self.hitpred_ctr > 3 else False
                break
        
        # Look for alternate hit
        for id in range(self.hitpred_id - 1, 0, -1):
            entry = self.tables[self.id2name[id]]['entries'][self.tage_idx[id]]
            if entry['tag'] == self.tage_tag[id]:
                if settings.DEBUG == 1:
                    print(f"ALT TAG MATCH FOUND ON predictor {id} :: idx {self.tage_idx[id]} : {self.tage_tag[id]}")
                if (self.use_alt_on_new_alloc < 0) or (entry['pred'] not in (3, 4)):
                    if settings.DEBUG == 1:
                        print("Not using alt :: entry is not newalloc")
                    self.altpred_id = id
                    self.altpred_ctr = entry['pred']
                    self.altpred_taken = True if self.altpred_ctr > 3 else False
                    break


            #isHit, pred = self.predict_tagged(self.id2name[id], self.branch_pc)
        
        if self.hitpred_id > 0:
            # if no altpred is found, use bimodal predictor's result
            if self.altpred_id > 0:
                if settings.DEBUG == 1:
                    print("ALTPRED IS FOUND IN TAGGED PREDICTORS")
                self.altpred_taken = True if self.altpred_ctr > 3 else False
            else:
                if settings.DEBUG == 1:
                    print("ALTPRED IS BIMODAL")
                self.altpred_taken = self.bim_pred
            
            if (self.use_alt_on_new_alloc < 0) or (self.hitpred_ctr not in (3,4)):
                if settings.DEBUG == 1:
                    print("TAGEPRED is MAIN HIT PREDICTOR")
                self.tage_pred = True if self.hitpred_ctr > 3 else False
            else:
                if settings.DEBUG == 1:
                    print("TAGEPRED IS ALTPRED")
                self.tage_pred = self.altpred_taken
        else:
            if settings.DEBUG == 1:
                print("NO HIT DETECTED. TAGE PRED == ALTPRED == BIMODAL")
            self.altpred_taken = self.bim_pred
            self.tage_pred = self.altpred_taken

        return self.tage_pred


    def update_bimodal(self, isTaken):
        idx_bimodal = (self.branch_pc >> self.word_align) & ((1<<self.base_idx_width) - 1)
        pred_b = self.tables['base']['pred'][idx_bimodal]
        #hyst_b = self.tables['base']['hyst'][idx_bimodal >> self.tables['base']['ent_hyst']]
        hyst_b = self.tables['base']['hyst'][idx_bimodal >> 2]

        (pred_b, hyst_b) = self.next_state_bimodal[(pred_b, hyst_b, isTaken)]

        self.tables['base']['pred'][idx_bimodal] = pred_b
        #self.tables['base']['hyst'][idx_bimodal >> self.tables['base']['ent_hyst']] = hyst_b
        self.tables['base']['hyst'][idx_bimodal >> 2] = hyst_b


    # TODO I NEED TO UPDATE FROM HIGHEST INDICIES TO LOWEST
    def update_ghist(self, isTaken):
        val = np.uint8(isTaken)
        self.ghist_ptr = self.ghist.head
        self.ghist.updateBuf(val)

        if settings.DEBUG == 1:
            print(f'ptr: {self.ghist_ptr}')
            print(f'GHIST: {self.ghist.getBuf(16)}')
        return
    
    def update_phist(self, branch_pc):
        phist = (int(self.phist) << 1) | ((branch_pc ^ (branch_pc >> 16)) & 1)
        
        #phist = (int(self.phist) << 1) | ((branch_pc) & 1)
        self.phist = np.uint16(phist & ((1<<self.phist_len) - 1))
        
        if settings.DEBUG == 1:
            print(hex(branch_pc))
            tmp = int(self.phist)
            binstr = ''
            for i in range(self.phist_len):
                #print(f'???{tmp& 1}')
                binstr = str(tmp&1) + binstr
                tmp >>= 1
            print(f'phist: 0b{binstr}')
        return
    
    def update_tage_ctr(self, isTaken, id):
        pred_name = self.id2name[id]
        ctr = self.tables[pred_name]['entries'][self.tage_idx[id]]['pred']

        if isTaken:
           if ctr < 7:
               self.tables[pred_name]['entries'][self.tage_idx[id]]['pred'] += 1
        else:
            if ctr > 0:
                self.tables[pred_name]['entries'][self.tage_idx[id]]['pred'] -= 1
        if settings.DEBUG == 1:
            foo = self.tables[pred_name]['entries'][self.tage_idx[id]]['pred']
            print(f'ctr update: {ctr} ->  {foo}')


    def update_tagged(self, isTaken):
        # Decide whether new entry should be allocated
        isNewAlloc = ((self.tage_pred != isTaken) and (self.hitpred_id < self.num_tagged))

        # if hitpred is not the base predictor
        if (self.hitpred_id > 0):
            hitpred_name = self.id2name[self.hitpred_id]
            assert(self.hitpred_ctr >= 0)
            #assert(self.altpred_ctr >= 0)

            hitpred_taken = self.hitpred_taken 
            
            foo  = True if self.hitpred_ctr > 3 else False
            assert (foo == self.hitpred_taken)
            
            altpred_taken = self.altpred_taken

            pseudoNewAlloc = True if (self.hitpred_ctr in (3,4)) else False

            # When new allocation is detected
            if(pseudoNewAlloc):
                if(hitpred_taken == isTaken):
                    isNewAlloc = False
                if(hitpred_taken != altpred_taken):
                    # if altpred got it right but hitpred didn't,
                    #   update use alt on new alloc counter
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

            for i in range(self.num_tagged, self.hitpred_id, -1):
                predname = self.id2name[i]
                if(self.tables[predname]['entries'][self.tage_idx[i]]['u'] < min_u):
                    min_u = self.tables[predname]['entries'][self.tage_idx[i]]['u']
            
            # find entry to allocate (select among 3 tables with longer history)
            # Randomly select a bank among the next longer history tables
            diff = int(self.num_tagged - self.hitpred_id)
            num_banks_to_consider = min(3, diff)
            random_offset = self.rand % num_banks_to_consider
            newentryId = self.hitpred_id + 1 + random_offset 
            
            # allocate new entry if none available
            if min_u > 0:
                self.tables[self.id2name[newentryId]]['entries'][self.tage_idx[newentryId]]['u'] = 0
            
            for i in range(newentryId, self.num_tagged + 1):
                #print(f'settings.DEBUG: {i} {newentryId}')
                if (self.tables[self.id2name[i]]['entries'][self.tage_idx[i]]['u'] == 0):
                    self.tables[self.id2name[i]]['entries'][self.tage_idx[i]]['tag'] = np.uint16(self.tage_tag[i])
                    self.tables[self.id2name[i]]['entries'][self.tage_idx[i]]['pred'] = np.uint8(4) if (isTaken) else np.uint8(3)
                    self.tables[self.id2name[i]]['entries'][self.tage_idx[i]]['u'] = np.uint8(0)
                    if settings.DEBUG == 1:
                        print(f'settings.DEBUG: {i} {newentryId}')
                        foo = self.tables[self.id2name[i]]['entries'][self.tage_idx[i]]['tag']
                        print(f'ENTRY CREATED at ID {i}, TAG {foo}')
                    break
        ## ALLOCATE DONE
        
        # RESET useful bit when tick is full 
        self.u_tick += 1
        if ((self.u_tick & ((1 << self.u_tick_log) - 1)) == 0):
            if settings.DEBUG == 1:
                print('RESETTING UBIT')
            for i in range(1, self.num_tagged + 1):
                for j in range(0, (1 << self.tables[self.id2name[i]]['ent_pred'])):
                    self.tables[self.id2name[i]]['entries'][j]['u'] >> 1
        # RESET ubit done

        # TODO: UPDATE COUNTERS
        if self.hitpred_id > 0:
            ## update the hit counter
            self.update_tage_ctr(isTaken, self.hitpred_id)
            if self.tables[self.id2name[self.hitpred_id]]['entries'][self.tage_idx[self.hitpred_id]]['u'] == 0:
                if self.altpred_id > 0:
                    self.update_tage_ctr(isTaken, self.altpred_id)
                elif self.altpred_id == 0:
                    # update bimodal
                    self.update_bimodal(isTaken)
        else:
            self.update_bimodal(isTaken)
        

        # TODO: UPDATE u counter
        if self.tage_pred != self.altpred_taken:
            if self.tage_pred == isTaken:
                if self.tables[self.id2name[self.hitpred_id]]['entries'][self.tage_idx[self.hitpred_id]]['u'] < 3:
                    self.tables[self.id2name[self.hitpred_id]]['entries'][self.tage_idx[self.hitpred_id]]['u'] += 1
                else:
                    if self.use_alt_on_new_alloc < 0:
                        if (self.tables[self.id2name[self.hitpred_id]]['entries'][self.tage_idx[self.hitpred_id]]['u'] > 0):
                            self.tables[self.id2name[self.hitpred_id]]['entries'][self.tage_idx[self.hitpred_id]]['u'] -= 1


        # update ghist and phist
        self.update_ghist(isTaken)
        self.update_phist(self.branch_pc)
        
        # update compressed histories
        for i in range(1, self.num_tagged + 1):
            name = self.id2name[i]
            
            #buf = self.ghist.getBuf(self.tables[name]['hist_len'] + 1)
            #self.comp_hist_idx[name].update(buf, self.ghist_ptr)
            #self.comp_hist_tag[0][name].update(buf, self.ghist_ptr)
            #self.comp_hist_tag[1][name].update(buf, self.ghist_ptr)

            self.comp_hist_idx[name].update2(self.ghist, 0, self.tables[name]['hist_len'])
            self.comp_hist_tag[0][name].update2(self.ghist, 0, self.tables[name]['hist_len'])
            self.comp_hist_tag[1][name].update2(self.ghist, 0, self.tables[name]['hist_len'])


        return 
    
    def train_predictor(self, isTaken):
        
        #print('UPDATE START------------')
        # USE LSFR for actual implementation
        #self.rand = random.randint(0, 3)
        self.rand = self.rand_array[self.rand_index]
        self.rand_index = (self.rand_index + 1) % len(self.rand_array)

        #self.rand = 2 if self.rand == 3 else self.rand
        # BIMODAL PREDICTOR
        #self.update_bimodal(isTaken)
        if settings.DEBUG == 1:
            print(f'RAND: {self.rand}\nISTAKEN: {isTaken}')
        self.update_tagged(isTaken)

        return 0