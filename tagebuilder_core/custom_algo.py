import numpy as np

def mix_path_history_adv(self, predictor_name:str, phist_size:int, phist:np.uint16)->int:
    """
    path history (history of a single bit from instruction address) hash generation
    for tagged entry access. A version similar to Seznec's submission to CBP
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

def custom_taggedComp_tag(self, predictor_name:str, bpc_hashed:int)->int:
    pass

def custom_taggedComp_idx(self, predictor_name:str, bpc_hashed:int)->int:
    pass