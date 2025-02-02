class helper:
    def __init__(self):
        self.br__type_enum = {"JMP": 0, "CALL": 1, "RET": 2}
        self.br__mode_enum = {"DIR": 0 , "IND": 1}
        self.br__cond_enum = {"UCD": 0, "CND": 1}

        self.br__type_unmap = ['JMP', 'CALL', 'RET']
        self.br__mode_unmap = ['DIR', 'IND']
        self.br__cond_unmap = ['UCD', 'CND']

    def encode_class(self, t, m, c)->int:
        """encod string class info to packed int

        Args:
            t (str): type
            m (str): mode 
            c (str): conditional
        """
        return self.br__type_enum[t]<<2 | self.br__mode_enum[m]<<1 | self.br__cond_enum[c]        
        
    def decode_class(self, cls_b):
        """decode packed class information to string

        Args:
            cls_b (_type_): packed int

        Returns:
            (str): set of decoded class strings 
        """
        t = self.br__type_unmap[(cls_b>>2) & 3]
        m = self.br__mode_unmap[(cls_b>>1) & 1]
        c = self.br__cond_unmap[(cls_b) & 1]
        return f'{t}+{m}+{c}' 
        