"""
BT9Reader Module

This module provides the BT9Reader class for reading and parsing BT9 formatted traces.
It supports metadata extraction, node and edge parsing, and batch processing of branch instructions.
"""

import cProfile
import gzip
import re
import numpy as np
import pandas as pd
import pstats

# Define numpy dtypes for trace nodes and edges

# Data type for nodes in the trace
node_dtype = np.dtype([
    ('vaddr', np.uint64),
    ('paddr', np.uint64),
    ('opcode', np.uint32),
    ('size', np.uint8),
    ('class', np.uint8)
    ])

# Data type for edges in the trace
edge_dtype = np.dtype([
    ('src_id', np.uint32),
    ('dest_id', np.uint32),
    ('taken', np.uint8),
    ('vaddr_target', np.uint64),
    ('paddr_target', np.uint64),
    ('inst_cnt', np.uint32)
    ])

# Data type for branch instructions used in batch processing
br_dtype = np.dtype([
    ('addr', np.uint64),
    ('taken', np.uint8),
    ('inst_cnt', np.uint16),
    #('class', np.uint8)
])

class BT9Reader:
    """
    Class for reading and parsing BT9 formatted traces.

    Attributes:
        filename (str): Path to the trace file.
        logger: Logger instance for logging information.
        file: File object for the trace.
        metadata (dict): Dictionary to store metadata.
        nodeArr (list): List to hold node information.
        edgeArr (list): List to hold edge information.
        addr_scoreboard_df (DataFrame): DataFrame for per-address scoreboard.
        predictor_scoreboard_df (DataFrame): DataFrame for predictor scoreboard.
        br_infoArr (ndarray): Array for batch parsed branch instructions.
        report (dict): Dictionary to hold simulation statistics.
        data (dict): Dictionary to store branch instruction count and performance metrics.
    """
    def __init__(self, filename, logger):
        # Debug and state tracking variables
        self.d = 0 # general debug counter
        self.d2 = None # previous edge id for verification
        self.d3 = 0 # sequence counter for verification
        self.disjoint = {} # store disjoint edges for debugging

        self.logger = logger
        self.filename = filename
        
        # Open the gzip file in text mode. TODO: handle errors gracefully.
        self.file = gzip.open(filename, mode="rt")
        self.br_seq_started = False
        
        self.metadata = {}
        self.nodeArr = [] # list to store nodes
        self.edgeArr = [] # list to store edges

        # DataFrames for tracking statistics
        self.addr_scoreboard_df = None
        self.predictor_scoreboard_df = None

        # Branch information placeholders (for single read -- DEPRECATED)
        self.br_addr = None
        self.br_taken = None
        self.br_inst_cnt = None

        self.br_infoArr = None # array for batch parsing of branch instructions

        # Report and simulation statistics
        self.report = {
            'is_sim_over': False,
            'total_instruction_count': 0,
            'branch_instruction_count': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'current_instruction_count': 0,
            'current_branch_instruction_count': 0,
            'current_accuracy': 0,
            'current_mpki': 0,
            'accuracy': 0,
            'mpki':0
        }
        
        # Data storage for metrics over time
        self.data = {
            'br_inst_cnt': [],
            'accuracy': [],
            'mpki': []
        }
        
        # Enumerations for branch types, modes, and conditions NOTE: ENUMS ARE MOVED TO HELPERS.PY -- clean this up
        self.br_type_enum = {"JMP": 0, "CALL": 1, "RET": 2}
        self.br_mode_enum = {"DIR": 0 , "IND": 1}
        self.br_cond_enum = {"UCD": 0, "CND": 1}
        
        # Reverse mappings for decoding
        self.br_type_unmap = ['JMP', 'CALL', 'RET']
        self.br_mode_unmap = ['DIR', 'IND']
        self.br_cond_unmap = ['UCD', 'CND']

        # Structure for predictor scoreboard columns
        self.predictor_scoreboard_structure = [
            'num_correct_preds',
            'num_incorrect_preds',
            'used_as_main',
            'used_as_alt',
            'conf_-4',
            'conf_-3',
            'conf_-2',
            'conf_-1',
            'conf_0',
            'conf_1',
            'conf_2',
            'conf_3'
        ]
    
    def init_predictor_scoreboard(self, num_predictors):
        """
        Initialize the predictor scoreboard DataFrame.

        Args:
            num_predictors (int): Number of predictors to initialize.
        """
        ids = [id for id in range(num_predictors)]
        self.predictor_scoreboard_df = pd.DataFrame({
            'num_correct_preds': 0,
            'num_incorrect_preds': 0,
            'used_as_main': 0,
            'used_as_alt': 0,
            'conf_-4': 0,
            'conf_-3': 0,
            'conf_-2': 0,
            'conf_-1': 0,
            'conf_0': 0,
            'conf_1': 0,
            'conf_2': 0,
            'conf_3': 0
        }, index = ids)
        self.predictor_scoreboard_df.index.name = 'pid'
    
    def read_metadata(self):
        """
        Read and parse metadata from the trace file.

        The metadata section starts with "BT9_SPA_TRACE_FORMAT" and ends before "BT9_NODES".
        """
        section = None
        for l in self.file:
            l = l.strip()

            if not l or l.startswith("#"):
                continue

            if l.startswith("BT9_SPA_TRACE_FORMAT"):
                section = "metadata"
                continue
            elif l.startswith("BT9_NODES"):
                break

            if section == "metadata":
                self._parse_metadata(l)

    def _parse_file(self):
        """
        Parse the trace file sections: metadata, nodes, and edges.

        Converts node and edge lists to numpy arrays and initializes the address scoreboard.
        """
        section = None
        for l in self.file:
            l = l.strip()

            if not l or l.startswith("#"):
                continue

            if l.startswith("BT9_SPA_TRACE_FORMAT"):
                section = "metadata"
                continue
            elif l.startswith("BT9_NODES"):
                section = "node"
                continue
            elif l.startswith("BT9_EDGES"):
                section = "edge"
                continue
            elif l.startswith("BT9_EDGE_SEQUENCE"):
                section = "edge_seq"
                self.br_seq_started = True
                if self.logger is not None:
                    self.logger.info(f'TRACE METADATA: {self.metadata}')
                # Convert lists to numpy arrays for efficiency
                self.nodeArr = np.array(self.nodeArr, node_dtype)
                self.edgeArr = np.array(self.edgeArr, edge_dtype)
                # Initialize per-address scoreboard using node virtual addresses and classes
                addrs = self.nodeArr['vaddr']
                classes = self.nodeArr['class']
                self.addr_scoreboard_df = pd.DataFrame({
                    'num_correct_preds': 0,
                    'num_incorrect_preds': 0,
                    'class': classes,
                    'trans': 0,
                    'prev_taken': 0
                }, index = addrs)
                self.addr_scoreboard_df.index.name = 'br_addr'
                self.logger.info(self.addr_scoreboard_df)
                self.report['total_instruction_count'] = int(self.metadata['total_instruction_count'])
                self.report['branch_instruction_count'] = int(self.metadata['branch_instruction_count'])
                break

            if section == "metadata":
                self._parse_metadata(l)
            elif section == "node":
                self._parse_node(l)
            elif section == "edge":
                self._parse_edge(l)
                


    def _parse_metadata(self, l):
        """
        Parse a single metadata line.

        Args:
            l (str): A line from the metadata section.
        """
        parts = re.split(r'[:\s]+', l)
        l_len = len(parts)
        if l_len == 0:
            assert(False)
        elif l_len == 1:
            self.metadata[parts[0]] = ""
        else:
            self.metadata[parts[0]] = parts[1]
            
    def encode_class(self, t, m, c)->int:
        """encod string class info to packed int NOTE: MOVED TO HELPERS.PY

        Args:
            t (str): type
            m (str): mode 
            c (str): conditional
        """
        return self.br_type_enum[t]<<2 | self.br_mode_enum[m]<<1 | self.br_cond_enum[c]        
        
    def decode_class(self, cls_b):
        """
        Decode a packed branch class integer into its components. NOTE: MOVED TO HELPERS.PY

        Args:
            cls_b (int): Packed branch class integer.

        Returns:
            tuple: (branch type, branch mode, branch condition) as strings.
        """
        t = self.br_type_unmap[(cls_b>>2) & 3]
        m = self.br_mode_unmap[(cls_b>>1) & 1]
        c = self.br_cond_unmap[(cls_b) & 1]
        return (t,m,c) 
    
    #    +->for idx 0
    #    |
    #[(vaddr, paddr, opcode, size), ... , (vaddr, paddr, opcode, size)]
    def _parse_node(self, l):
        """
        Parse a node line and add it to the node array.

        Args:
            l (str): A line from the node section.
        """
        p = l.split()
        if len(p) > 6:
            # Parse branch class if available
            cls_str = p[7]
            cls_str = cls_str.split('+')
            cls = self.encode_class(*cls_str)
        else:
            cls = 0
        
        el = (
            np.uint64(int(p[2], 16)),   # virtual address
            np.uint64(0) if p[3] == "-" else np.uint64(int(p[3], 16)), # physical address
            np.uint32(int(p[4], 16)),   # opcode
            np.uint8(p[5]),             # size
            np.uint8(cls)               # encoded branch class
        )
        self.nodeArr.append(el)
        # Validate node order consistency
        assert(p[0] == "NODE")
        assert(self.nodeArr[int(p[1])] == el and self.nodeArr[int(p[1])] is el)

    def _parse_edge(self, l):
        """
        Parse an edge line and add it to the edge array.

        Args:
            l (str): A line from the edge section.
        """
        p = l.split()
        el = (
            np.uint32(int(p[2])), # src node id
            np.uint32(int(p[3])), # dest node id
            np.uint8(1) if p[4] == 'T' else np.uint8(0), # taken flag
            np.uint64(int(p[5], 16)), # target virtual address
            np.uint64(0) if p[6] == "-" else np.uint64(int(p[6], 16)), # target physical address
            np.uint32(int(p[7])) # instruction count between nodes
            )
        self.edgeArr.append(el)
        # Update debug counter using weighted instruction count
        self.d += int(p[9])*(1+int(p[7]))
    
    def init_tables(self):
        """
        Initialize parsing tables by processing the file.
        """
        self._parse_file()
    
    def verify_graph(self):
        """
        Verify the graph structure of the parsed trace.

        Ensures that the destination node of the previous edge matches
        the source node of the current edge.

        Returns:
            int: 1 if EOF is detected, 0 otherwise.
        """
        assert(self.br_seq_started)
        l = self.file.readline()
        p = l.strip()
        if p.startswith('EOF'):
            return 1
        edge_id = int(p)
        if self.d2:
           if (self.edgeArr[self.d2]['dest_id'] != self.edgeArr[edge_id]['src_id']):
                if p not in self.disjoint.keys():
                    self.disjoint[p] = (self.d2, self.edgeArr[self.d2], edge_id, self.edgeArr[edge_id])
                # Disjoint edges found; additional error handling can be added here.
        self.br_addr = self.nodeArr[int(self.edgeArr[edge_id]['src_id'])]['vaddr']
        self.br_taken = self.edgeArr[edge_id]['taken']
        self.br_inst_cnt = self.edgeArr[edge_id]['inst_cnt']
        self.d2 = edge_id # update previous edge id
        return 0


    def read_branch(self):
        """
        Read a single branch instruction from the trace. NOTE NOT USED ANYMORE FOR OPTIMIZATION PURPOSES

        Returns:
            int: -1 for incomplete file, 1 for EOF, 0 for successful read.
        """
        assert(self.br_seq_started)
        l = self.file.readline()
        if l == '':
            return -1   # incomplete file detected
        
        p = l.split()
        if len(p) > 1 and p[0] == 'EOF':
            print(p[0])
            return 1    # EOF detected

        edge_id = int(p[0])
        self.br_addr = self.nodeArr[int(self.edgeArr[edge_id]['src_id'])]['vaddr']
        self.br_taken = self.edgeArr[edge_id]['taken']
        self.br_inst_cnt = self.edgeArr[edge_id]['inst_cnt']
        return 0

    def read_branch_batch(self, size = 10_000):
        """
        Read a batch of branch instructions from the trace.

        Args:
            size (int): Number of lines to read in a batch (default is 10,000).

        Returns:
            int: -1 for incomplete file, 1 if EOF reached, 0 for successful batch read.
        """
        retflag = 0
        assert(self.br_seq_started)
        ls = []
        for _ in range(size):
            ls.append(self.file.readline())
        #ls = self.file.readlines(size)
        #print("what",len(ls))

        if not ls:
            return -1

        edges_ids = []
        for l in ls:
            l = l.strip()
            if not l:
                assert(False) # unexpected empty line
            if l.startswith('EOF'):
                if self.logger is not None:
                    self.logger.info('EOF')
                retflag = 1
                break
            else:
                edges_ids.append(int(l))
        # Convert edge ids to a numpy array for vectorized processing
        edges_ids = np.array(edges_ids, dtype=np.uint32)
        edges = self.edgeArr[edges_ids]
        src_ids = edges['src_id']
        # Retrieve branch addresses from nodes
        br_addrs = np.array(self.nodeArr[src_ids]['vaddr'], dtype=np.uint64)
        br_takens = np.array(edges['taken'], dtype = np.uint8)
        br_inst_cnts = np.array(edges['inst_cnt'], dtype = np.uint16)
        
        # Prepare branch information array for batch processing
        self.br_infoArr = np.zeros(len(br_addrs), dtype=br_dtype)
        self.br_infoArr['addr'] = br_addrs
        self.br_infoArr['taken'] = br_takens
        self.br_infoArr['inst_cnt'] = br_inst_cnts

        return retflag

    def update_stats(self, ismatching):
        """
        Update simulation statistics based on prediction outcome.

        Args:
            ismatching (bool): True if prediction was correct, False otherwise.
        """
        if ismatching:
            self.report['correct_predictions'] += 1
        else:
            self.report['incorrect_predictions'] += 1
    
    def finalize_stats(self):
        """
        Finalize statistics by calculating overall accuracy and MPKI.
        """
        self.report['accuracy'] = self.report['correct_predictions'] / self.report['branch_instruction_count']
        self.report['mpki'] = 1000 * self.report['incorrect_predictions'] / self.report['total_instruction_count']
    
    def close(self):
        """
        Close the trace file.
        """
        self.file.close()

    def __del__(self):
        """
        Destructor to ensure the file is closed.
        """
        self.close()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    profiler.disable()
    file = "/Users/wonjongbot/tageBuilder/CBP16 Data/evaluationTraces/SHORT_MOBILE-1.bt9.trace.gz"
    r = BT9Reader(file)

    r.init_tables()

    # NONE BATCH IMPLEMENTATION
    # while True:
    #     ret = r.read_branch()
    #     if ret == -1:
    #         print('INCOMPLETE FILE DETECTED')
    #         break
    #     elif ret == 1:
    #         break
    #     r.report['current_branch_instruction_count'] += 1
    #     r.report['current_instruction_count'] += (1 + int(r.br_inst_cnt))

    # BATCH IMPLEMENTATION
    while True:
        ret = r.read_branch_batch()
        if ret == -1:
            print('INCOMPLETE FILE DETECTED')
            break
        for b in r.br_infoArr:                
            r.report['current_branch_instruction_count'] += 1
            r.report['current_instruction_count'] += (1 + int(b['inst_cnt']))
        if ret == 1:
            #r.report['current_branch_instruction_count'] += 1
            r.report['is_sim_over'] = True
            break

    # for checking graph structure
    # r.d3 = 0
    # while True:
    #     ret = r.verify_graph()
    #     if ret == 1:
    #         print('EOF')
    #         break
    #     else:
    #         r.d3 += 1
    #         r.report['current_branch_instruction_count'] += 1
    #         r.report['current_instruction_count'] += (1 + int(r.br_inst_cnt))
    #         continue
    # print('disjoint')
    # for k,v in r.disjoint.items():
    #     t0 = v[1]['src_id']
    #     t1 = v[1]['dest_id']
    #     t2 = v[3]['src_id']
    #     t3 = v[3]['dest_id']
    #     print(f'    {k}: {v[0]} :: {t0} {t1} | {v[2]} :: {t2} {t3}')

    print('Calculated from edge list')
    print(f'    {r.d}')
    print('REPORT')
    for k, v in r.report.items():
        print(f'    {k}: {v}')
    print('META')
    for k, v in r.metadata.items():
        print(f'    {k}: {v}')

    profiler.disable()
    with open(f"nBatch_profiler.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats("cumulative")  # Sort by cumulative time
        stats.print_stats()