import cProfile
import gzip
import re
import numpy as np
import pstats
from collections import defaultdict


node_dtype = np.dtype([
    ('vaddr', np.uint64),
    ('paddr', np.uint64),
    ('opcode', np.uint32),
    ('size', np.uint8)
    ])

edge_dtype = np.dtype([
    ('src_id', np.uint32),
    ('dest_id', np.uint32),
    ('taken', np.uint8),
    ('vaddr_target', np.uint64),
    ('paddr_target', np.uint64),
    ('inst_cnt', np.uint32)
    ])

br_dtype = np.dtype([
    ('addr', np.uint64),
    ('taken', np.uint8),
    ('inst_cnt', np.uint16)
])

# BT9Reader class that supports BT9 formatted traces
class BT9Reader:
    def __init__(self, filename, logger):
        self.d = 0 # debug purposes
        self.d2 = None
        self.d3 = 0
        self.disjoint = {}

        self.logger = logger

        self.filename = filename
        self.file = gzip.open(filename, mode="rt") #TODO: handle error gracefully
        self.br_seq_started = False
        
        self.metadata = {}
        self.nodeArr = []
        self.edgeArr = []

        # scoreboard = { vaddr: {num_exe: 0, num_incorrect_preds: 0} }
        self.addr_scoreboard = defaultdict(lambda: {'num_correct_preds': 0, 'num_incorrect_preds': 0})

        self.br_addr = None
        self.br_taken = None
        self.br_inst_cnt = None

        self.br_infoArr = None # for batch parsing

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

        self.data = {
            'br_inst_cnt': [],
            'accuracy': [],
            'mpki': []
        }
    
    def read_metadata(self):
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
                self.nodeArr = np.array(self.nodeArr, node_dtype)
                self.edgeArr = np.array(self.edgeArr, edge_dtype)
                if self.logger is not None:
                    self.logger.info(f'{self.d}')
                self.report['total_instruction_count'] = int(self.metadata['total_instruction_count'])
                self.report['branch_instruction_count'] = int(self.metadata['branch_instruction_count'])
                # while True:
                #     continue
                # for i in range(2588 - 10, 2588 + 1):
                #     print(f'Node {i}')
                #     for k in node_dtype.names:
                #         print(f'    {k}: {hex(self.nodeArr[i][k])}')

                # for i in range(3081 - 10, 3081 + 1):
                #     print(f'Edge {i}')
                #     for k in edge_dtype.names:
                #         print(f'    {k}: {self.edgeArr[i][k]} <-> {hex(self.edgeArr[i][k])}')
                break

            if section == "metadata":
                self._parse_metadata(l)
            elif section == "node":
                self._parse_node(l)
            elif section == "edge":
                self._parse_edge(l)
                


    def _parse_metadata(self, l):
        parts = re.split(r'[:\s]+', l)
        l_len = len(parts)
        if l_len == 0:
            assert(False)
        elif l_len == 1:
            self.metadata[parts[0]] = ""
        else:
            self.metadata[parts[0]] = parts[1]

    #    +->for idx 0
    #    |
    #[(vaddr, paddr, opcode, size), ... , (vaddr, paddr, opcode, size)]
    def _parse_node(self, l):
        p = l.split()
        el = (
            np.uint64(int(p[2], 16)), # vaddr
            np.uint64(0) if p[3] == "-" else np.uint64(int(p[3], 16)), # paddr
            np.uint32(int(p[4], 16)), # opcode
            np.uint8(p[5]) # size
            )
        self.nodeArr.append(el)

        assert(p[0] == "NODE")
        assert(self.nodeArr[int(p[1])] == el and self.nodeArr[int(p[1])] is el)

    def _parse_edge(self, l):
        p = l.split()
        el = (
            np.uint32(int(p[2])), # src id
            np.uint32(int(p[3])), # dest id
            np.uint8(1) if p[4] == 'T' else np.uint8(0), # taken
            np.uint64(int(p[5], 16)), # vaddr target
            np.uint64(0) if p[6] == "-" else np.uint64(int(p[6], 16)), #paddr target
            np.uint32(int(p[7])) # inst cnt
            )
        self.edgeArr.append(el)
        #print(p)
        self.d += int(p[9])*(1+int(p[7]))
    
    def init_tables(self):
        self._parse_file()
    
    def verify_graph(self):
        """
        used to verify graph structure and weird quirks of the traces
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
                #print(f'LINECONTENT: {p}')
                #print(f'SEQ # {self.d3}\nEDGE # {self.d2}::{self.edgeArr[self.d2]}\nEDGE # {edge_id}::{self.edgeArr[edge_id]}')
                #assert(self.edgeArr[self.d2]['dest_id'] == self.edgeArr[edge_id]['src_id'])
                #return 2
        self.br_addr = self.nodeArr[int(self.edgeArr[edge_id]['src_id'])]['vaddr']
        self.br_taken = self.edgeArr[edge_id]['taken']
        self.br_inst_cnt = self.edgeArr[edge_id]['inst_cnt']
        self.d2 = edge_id# previous edge
        return 0


    def read_branch(self):
        assert(self.br_seq_started)
        l = self.file.readline()
        if l == '':
            return -1 # detect incomplete files
        
        p = l.split()
        if len(p) > 1 and p[0] == 'EOF':
            print(p[0])
            return 1 # successfully detected EOF

        edge_id = int(p[0])
        self.br_addr = self.nodeArr[int(self.edgeArr[edge_id]['src_id'])]['vaddr']
        self.br_taken = self.edgeArr[edge_id]['taken']
        self.br_inst_cnt = self.edgeArr[edge_id]['inst_cnt']

        
        return 0

    def read_branch_batch(self, size = 10_000):
        retflag = 0
        assert(self.br_seq_started)
        ls = []
        for i in range(size):
            ls.append(self.file.readline())
        #ls = self.file.readlines(size)
        #print("what",len(ls))

        if not ls:
            return -1

        edges_ids = []
        for l in ls:
            l = l.strip()
            if not l:
                assert(False) # this probably shouldn't happen
            if l.startswith('EOF'):
                if self.logger is not None:
                    self.logger.info('EOF')
                retflag = 1
                break
            else:
                edges_ids.append(int(l))
        #print(edges_ids)
        
        edges_ids = np.array(edges_ids, dtype=np.uint32)

        edges = self.edgeArr[edges_ids]
        src_ids = edges['src_id']
        br_addrs = np.array(self.nodeArr[src_ids]['vaddr'], dtype=np.uint64)
        br_takens = np.array(edges['taken'], dtype = np.uint8)
        br_inst_cnts = np.array(edges['inst_cnt'], dtype = np.uint16)
        #print(len(br_addrs))

        self.br_infoArr = np.zeros(len(br_addrs), dtype=br_dtype)

        self.br_infoArr['addr'] = br_addrs
        self.br_infoArr['taken'] = br_takens
        self.br_infoArr['inst_cnt'] = br_inst_cnts

        return retflag

    def update_stats(self, ismatching):
        if ismatching:
            self.report['correct_predictions'] += 1
        else:
            self.report['incorrect_predictions'] += 1
    
    def finalize_stats(self):
        self.report['accuracy'] = self.report['correct_predictions'] / self.report['branch_instruction_count']
        self.report['mpki'] = 1000 * self.report['incorrect_predictions'] / self.report['total_instruction_count']
    
    def close(self):
        self.file.close()

    def __del__(self):
        self.close()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    profiler.disable()
    file = "/home/wonjongbot/tageBuilder/cbp16/traces/SHORT_MOBILE-24.bt9.trace.gz"
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