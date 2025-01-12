import yaml
from tagebuilder_core import bt9reader
from tagebuilder_core import tage_predictor
from tagebuilder_core import settings
from tagebuilder_core import tage_optimized
from numba import njit

import numpy as np
import json
import time
import resource
import logging
import argparse

import cProfile
import pstats

from datetime import datetime

settings.READ_BATCH = True
# Get the current time
current_time = datetime.now()

# Format the time as a string suitable for file names
file_name_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

configname = settings.SPEC_NAME

# the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{settings.REPORT_DIR}/logger/logger_{configname}_{file_name_time}.log"),
    ],
)

ITER_1k =   1000
ITER_10k =  10_000
ITER_100k = 100_000
ITER_1M =   1_000_000
ITER_10M =  10_000_000
ITER_30M =  30_000_000
ITER_60M =  60_000_000
ITER_100M = 100_000_000
ITER_300M = 300_000_000
ITER_600M = 600_000_000
ITER_1B =   1_000_000_000
ITER_10B =  10_000_000_000
iterflags = (
        ITER_1k,
        ITER_10k,
        ITER_100k,
        ITER_1M,
        ITER_10M,
        ITER_30M,
        ITER_60M,
        ITER_100M,
        ITER_300M,
        ITER_600M,
        ITER_1B,
        ITER_10B
)
    

def updateMPKI(reader):
    curr_inst_cnt = reader.report['current_instruction_count'] + 1
    curr_br_cnt = reader.report['current_branch_instruction_count'] + 1
    reader.report['current_accuracy'] = reader.report['correct_predictions'] / curr_br_cnt
    reader.report['current_mpki'] = 1000 * reader.report['incorrect_predictions'] / curr_inst_cnt

# cool way to show stats -- inspired by CBP16 eval
def statHeartBeat(reader):
    iter = reader.report['current_branch_instruction_count']
    if iter in iterflags:
        updateMPKI(reader)
        progressout = 'PROGRESS\n'
        for i in ('current_instruction_count', 'current_branch_instruction_count', 'current_accuracy', 'current_mpki'):
            progressout += f'    {i}: {reader.report[i]}'
        print(progressout)
        


def main(NUM_INSTR = -1, spec_name = "tage_custom.json"):
    mainlogger = logging.getLogger(f"{__name__}")
    memout = ''
    out = ''
    sum_acc = 0
    sum_mpki = 0
    current_time = time.time()
    last_progress_time = current_time
    filelist = [
        #('SHORT_MOBILE-24', settings.CBP16_TRACE_DIR + 'SHORT_MOBILE-24.bt9.trace.gz'),
        # ('SHORT_MOBILE-1', settings.CBP16_TRACE_DIR + 'SHORT_MOBILE-1.bt9.trace.gz')
        ('SHORT_SERVER-1', settings.CBP16_TRACE_DIR + 'SHORT_SERVER-1.bt9.trace.gz')
    ]

    mainlogger.info('Tested traces:\n'+'\n'.join([f"('{name}', {path})" for name, path in filelist]))
    
    with open(spec_name, 'r') as f:
        spec = json.load(f)

    predictor = tage_predictor.TAGEPredictor()
    predictor.init_tables(spec)
    memout += predictor.sizelog

    for bm in filelist:
        print(f'TESTING {bm[0]}')
        reader = bt9reader.BT9Reader(bm[1])
        reader.init_tables()
        while True:
            current_time = time.time()
            # read batch by default
            b_size = 1000
            result = reader.read_branch_batch(b_size)
            if result == -1:
                print('INCOMPLETE FILE DETECTED')
                break
            #print(reader.br_infoArr)
            for b in reader.br_infoArr:
                predictor.branch_pc = b['addr']
                taken = bool(b['taken'])
                pred = predictor.make_prediction()
                reader.update_stats(pred == taken)
                predictor.train_predictor(taken)
                reader.report['current_branch_instruction_count'] += 1
                reader.report['current_instruction_count'] += (1 + int(b['inst_cnt']))
            statHeartBeat(reader)
            if result == 1:
                #reader.report['current_branch_instruction_count'] += 1
                reader.report['is_sim_over'] = True
                break
            
        assert(reader.report['is_sim_over'])
        
        reader.finalize_stats()
        out += 'REPORT\n'
        out += f'   TEST: {bm[0]}\n'
        for k,v in reader.report.items():
            out += f'    {k}: {v}\n'
        out += memout
        #print(out_f)
    return out

def main_optimized_tage(NUM_INSTR = -1, spec_name = "tage_custom.json"):
    mainlogger = logging.getLogger(f"{__name__}")
    out = ''
    filelist = [
        # ('SHORT_MOBILE-24', settings.CBP16_TRACE_DIR + 'SHORT_MOBILE-24.bt9.trace.gz'),
        # ('SHORT_MOBILE-1', settings.CBP16_TRACE_DIR + 'SHORT_MOBILE-1.bt9.trace.gz'),
        # ('LONG_MOBILE-1', settings.CBP16_TRACE_DIR + 'LONG_MOBILE-1.bt9.trace.gz'),
        ('SHORT_SERVER-1', settings.CBP16_TRACE_DIR + 'SHORT_SERVER-1.bt9.trace.gz'),
        # ('LONG_SERVER-1', settings.CBP16_TRACE_DIR + 'LONG_SERVER-1.bt9.trace.gz')
    ]

    mainlogger.info('Tested traces:\n'+'\n'.join([f"('{name}', {path})" for name, path in filelist]))
    
    with open(spec_name, 'r') as f:
        spec = yaml.safe_load(f)

    predictor = tage_optimized.TAGEPredictor(spec)

    out += f'STORAGE REPORT'
    for k,v in predictor.storage_report.items():
        out += f"    {k}: {v}\n"

    for bm in filelist:
        print(f'TESTING {bm[0]}')
        reader = bt9reader.BT9Reader(bm[1])
        reader.init_tables()
        while True:
            current_time = time.time()
            # read batch by default
            b_size = 1000
            result = reader.read_branch_batch(b_size)
            if result == -1:
                print('INCOMPLETE FILE DETECTED')
                break
            
            #print(reader.br_infoArr)
            results = tage_optimized.make_pred_n_update_batch(
                reader.br_infoArr,
                predictor.num_tables,
                predictor.base_entries,
                predictor.tagged_entries,
                predictor.tagged_offsets, #tagged_offsets
                predictor.tagged_idxs,
                predictor.tagged_tags,
                predictor.tagged_num_entries_log,
                predictor.hist_len_arr, #hist_len_arr
                predictor.comp_hist_idx_arr, #comp_hist_idx_arr
                predictor.comp_hist_tag0_arr, #comp_hist_tag0_arr
                predictor.comp_hist_tag1_arr, #comp_hist_tag1_arr
                predictor.tagged_tag_widths, #tagged_tag_widths
                predictor.ghist,
                #predictor.phist, #phist
                #predictor.use_alt_on_new_alloc, #use_alt_on_new_alloc
                predictor.metadata[0], #metadata
                predictor.rand_array
                )
            
            for i, r in enumerate(results):
                reader.update_stats(bool(r))
                reader.report['current_branch_instruction_count'] += 1
                reader.report['current_instruction_count'] += (1 + int(reader.br_infoArr[i]['inst_cnt']))
            
            statHeartBeat(reader)
            if result == 1:
                #reader.report['current_branch_instruction_count'] += 1
                reader.report['is_sim_over'] = True
                break
            
        assert(reader.report['is_sim_over'])
        
        reader.finalize_stats()
        out += 'Sim report\n'
        out += f'    Sim name: {bm[0]}\n'
        for k,v in reader.report.items():
            out += f'    {k}: {v}\n'
        #print(out_f)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tagebuilder!")
    parser.add_argument("-spec", type=str, help="spec name")
    parser.add_argument("-o", "--optimized", action="store_true", help="Use optimzed tage sim")
    
    args = parser.parse_args()
    print(args)
    spec = args.spec
    optimized = args.optimized


    if not optimized:
        with open(f'{settings.REPORT_DIR}UNOPTIMIZED_{spec}_{file_name_time}.txt', 'w') as f:
            #profiler = cProfile.Profile()
            #profiler.enable()
            start_wall = time.time()
            start_resources = resource.getrusage(resource.RUSAGE_SELF)

            out = main(NUM_INSTR = -1, spec_name= settings.SPEC_DIR+spec+".json")
            
            end_wall = time.time()
            end_resources = resource.getrusage(resource.RUSAGE_SELF)

            real_time = end_wall - start_wall
            user_time = end_resources.ru_utime - start_resources.ru_utime
            sys_time  = end_resources.ru_stime - start_resources.ru_stime
            time_str = f'\nTIME\n'
            time_str += f'    real {real_time:.3f} s\n'
            time_str += f'    user {user_time:.3f} s\n'
            time_str += f'    sys  {sys_time:.3f} s\n'

            out += time_str
            #profiler.disable()

            f.write(out)
    else:
        with open(f'{settings.REPORT_DIR}OPTIMIZED_{spec}_{file_name_time}.txt', 'w') as f:
            #profiler = cProfile.Profile()
            #profiler.enable()

            start_wall = time.time()
            start_resources = resource.getrusage(resource.RUSAGE_SELF)
            
            out = main_optimized_tage(NUM_INSTR = -1, spec_name= settings.SPEC_DIR+spec+".yaml")
            
            end_wall = time.time()
            end_resources = resource.getrusage(resource.RUSAGE_SELF)

            real_time = end_wall - start_wall
            user_time = end_resources.ru_utime - start_resources.ru_utime
            sys_time  = end_resources.ru_stime - start_resources.ru_stime
            time_str = f'\nTIME\n'
            time_str += f'    real {real_time:.3f} s\n'
            time_str += f'    user {user_time:.3f} s\n'
            time_str += f'    sys  {sys_time:.3f} s\n'

            out += time_str
            #profiler.disable()

            f.write(out)

    #with open(f"{settings.REPORT_DIR}profiled/OPTIMIZED_profile_results_{configname}_{file_name_time}.txt", "w") as f:
    #  stats = pstats.Stats(profiler, stream=f)
    #  stats.sort_stats("cumulative")  # Sort by cumulative time
    #  stats.print_stats()