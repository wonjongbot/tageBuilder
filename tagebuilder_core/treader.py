import struct
import numpy as np 
import os
import logging

class TraceReader:
    def __init__(self, filename):
        """
        Initialize the class by
            * opening the binary trace file
            * initialize variables for statistics
                * stat_num_instr: total instruction count
                * stat_num_br: total branch instruction count
                * stat_mis_pred: total mispredicted branch instruction count
                * stat_accuracy: prediction %
                * stat_mispKI: mispredictions/1000s of instructions
            * initialize variables to interface with predictor model
                * instr_addr: instruction address
                * br_taken: whether branch at given instr_addr is taken
        """

        self.logger = logging.getLogger(f"{__name__}")
        self.logger.info("Initializing TraceReader")

        self.filename = filename
        self.file = open(filename, 'rb')
        self.stat_num_instr = 0
        self.stat_num_br_est = 0
        self.stat_num_br = 0
        self.stat_mis_pred = 0
        self.stat_accuracy = 0
        self.stat_mispKI = 0
        self.read_stats()
        
        self.instr_addr = None
        self.br_taken = None

        # [(<instr_addr>, <br_taken>) ... ]
        self.br_info_arr = []
    def read_stats(self):
        """
        Read total number of instructions from the first 4 bytes of the trace file
        """
        data = self.file.read(4)
        if len(data) < 4:
            raise ValueError("File too small, cannot read initial 4 bytes.")
        # Convert from big endian to host endianness (little)
        self.stat_num_instr = struct.unpack('>I', data)[0]
        self.stat_num_br_est = (os.stat(self.filename).st_size - 4) // 5

    def read_branch(self):
        """
        Read the next 5 bytes from the file. The first 4 bytes are interpreted
        as instruction address. The 5th byte is branch result.
        
        Returns:
            int: 0 on success, -1 on failure (e.g., end of file).
        """
        data = self.file.read(5)
        if len(data) < 5:
            return -1 
        self.instr_addr = struct.unpack('<I', data[:4])[0]
        self.br_taken = bool(data[4])
        self.stat_num_br+=1
        return 0
    
    def read_branch_batch(self, batch_size = 1024):
        """
        Read branches in a block of (by default) 1024 branch instructions then return as list

        Returns:
            []: [str] on full/partial block, [] on empty block (EOF)
        """
        self.br_info_arr = []
        data = self.file.read(5*batch_size)
        if data == b'':
            self.logger.info("EOF detected")
            return -1
        
        self.stat_num_br += (len(data) // 5)
        for i in range(len(data) // 5):
            self.br_info_arr.append((struct.unpack('<I', data[5*i:5*i+4])[0], bool(data[5*i + 4])))
            #print(self.br_info_arr[-1])
        return 0

        
    
    def update_stats(self, outcome):
        if(outcome != self.br_taken):
            self.stat_mis_pred += 1

    def report_stats(self):
        """
        Return a report string for the benchmark
        
        Returns:
            str: The report string.
        """
        report = f"Number of instructions:\t\t{self.stat_num_instr}\n"
        report += f"Number of branches:\t\t{self.stat_num_br}\n"
        report += f"Number of mispredictions:\t{self.stat_mis_pred}\n"
        self.stat_accuracy = 100*(1 - self.stat_mis_pred / self.stat_num_br)
        report += f"prediction %:\t{self.stat_accuracy:,.3f}\n"
        self.stat_mispKI = self.stat_mis_pred / (self.stat_num_instr/1000)
        report += f"1000*wrong_cc_predicts/total insts => 1000 * {self.stat_mis_pred} / {self.stat_num_instr} = {self.stat_mispKI:,.3f}\n"
        return report
    
    def get_stats(self):
        """
        get prediction % and misp/KI
        """
        return (self.stat_accuracy, self.stat_mispKI)

    def __del__(self):
        """
        Close the file when done to free up system resources.
        """
        self.file.close()