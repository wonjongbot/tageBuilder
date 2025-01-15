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
import pandas as pd
import matplotlib.pyplot as plt

import os
import cProfile
import pstats

from datetime import datetime


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
    curr_inst_cnt = reader.report['current_instruction_count']
    curr_br_cnt = reader.report['current_branch_instruction_count']
    reader.report['current_accuracy'] = reader.report['correct_predictions'] / curr_br_cnt
    reader.report['current_mpki'] = 1000 * reader.report['incorrect_predictions'] / curr_inst_cnt


# cool way to show stats -- inspired by CBP16 eval
#   modified to store dataframe for visualization
def statHeartBeat(reader):
    iter = reader.report['current_branch_instruction_count']
    updateMPKI(reader)
    reader.data['br_inst_cnt'].append(reader.report['current_branch_instruction_count'])
    reader.data['accuracy'].append(reader.report['current_accuracy'])
    reader.data['mpki'].append(reader.report['current_mpki'])
    if iter in iterflags:
        #updateMPKI(reader)
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
        ('SHORT_MOBILE-1', settings.CBP16_TRACE_DIR + 'SHORT_MOBILE-1.bt9.trace.gz'),
        ('SHORT_MOBILE-2', settings.CBP16_TRACE_DIR + 'SHORT_MOBILE-2.bt9.trace.gz'),
        ('SHORT_MOBILE-3', settings.CBP16_TRACE_DIR + 'SHORT_MOBILE-3.bt9.trace.gz'),
        ('SHORT_MOBILE-4', settings.CBP16_TRACE_DIR + 'SHORT_MOBILE-4.bt9.trace.gz'),
        ('LONG_MOBILE-1', settings.CBP16_TRACE_DIR + 'LONG_MOBILE-1.bt9.trace.gz'),
        ('LONG_MOBILE-2', settings.CBP16_TRACE_DIR + 'LONG_MOBILE-2.bt9.trace.gz'),
        ('LONG_MOBILE-3', settings.CBP16_TRACE_DIR + 'LONG_MOBILE-3.bt9.trace.gz'),
        ('LONG_MOBILE-4', settings.CBP16_TRACE_DIR + 'LONG_MOBILE-4.bt9.trace.gz'),
        ('SHORT_SERVER-1', settings.CBP16_TRACE_DIR + 'SHORT_SERVER-1.bt9.trace.gz'),
        ('SHORT_SERVER-2', settings.CBP16_TRACE_DIR + 'SHORT_SERVER-2.bt9.trace.gz'),
        ('SHORT_SERVER-3', settings.CBP16_TRACE_DIR + 'SHORT_SERVER-3.bt9.trace.gz'),
        ('SHORT_SERVER-4', settings.CBP16_TRACE_DIR + 'SHORT_SERVER-4.bt9.trace.gz'),
        ('LONG_SERVER-1', settings.CBP16_TRACE_DIR + 'LONG_SERVER-1.bt9.trace.gz'),
        ('LONG_SERVER-2', settings.CBP16_TRACE_DIR + 'LONG_SERVER-2.bt9.trace.gz'),
        ('LONG_SERVER-3', settings.CBP16_TRACE_DIR + 'LONG_SERVER-3.bt9.trace.gz'),
        ('LONG_SERVER-4', settings.CBP16_TRACE_DIR + 'LONG_SERVER-4.bt9.trace.gz')
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

def run_single_sim(spec_name = "tage_sc_l", test_name = "SHORT-MOBILE-1"):
    out = {}
    fileinfo = (test_name, settings.CBP16_TRACE_DIR + test_name + '.bt9.trace.gz')

    start_wall = time.time()
    start_resources = resource.getrusage(resource.RUSAGE_SELF)

    out['sim_id'] = 0
    out['timestamp'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    out['config'] = {}
    out['config']['predictor'] = spec_name
    out['config']['spec_file_dir'] = settings.SPEC_DIR + spec_name + '.yaml'
    out['config']['trace'] = fileinfo[0]
    out['config']['trace_file_dir'] = fileinfo[1]
    
    with open(out['config']['spec_file_dir'], 'r') as f:
        spec = yaml.safe_load(f)

    predictor = tage_optimized.TAGEPredictor(spec)

    out['storage_report'] = {}
    for k,v in predictor.storage_report.items():
        out['storage_report'][k] = v

    print(f'TESTING {fileinfo[0]} :: {fileinfo[1]}')
    reader = bt9reader.BT9Reader(fileinfo[1])
    reader.init_tables()
    while True:
        # read batch by default
        b_size = 10000
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
            predictor.metadata[0], #metadata
            predictor.rand_array
            )
        
        for i, r in enumerate(results):
            # reader.update_stats(bool(r)) # remove function call overhead
            if bool(r):
                reader.report['correct_predictions'] += 1
            else:
                reader.report['incorrect_predictions'] += 1
            reader.report['current_branch_instruction_count'] += 1
            reader.report['current_instruction_count'] += (1 + int(reader.br_infoArr[i]['inst_cnt']))
        
        statHeartBeat(reader)
        if result == 1:
            #reader.report['current_branch_instruction_count'] += 1
            reader.report['is_sim_over'] = True
            break
    
    assert(reader.report['is_sim_over'])

    reader.finalize_stats()

    end_wall = time.time()
    end_resources = resource.getrusage(resource.RUSAGE_SELF)

    real_time = end_wall - start_wall
    user_time = end_resources.ru_utime - start_resources.ru_utime
    sys_time  = end_resources.ru_stime - start_resources.ru_stime
    time_report = {}
    time_report['real'] = real_time
    time_report['user'] = user_time
    time_report['sys'] = sys_time


    out['perf_report'] = {}
    for k,v in reader.report.items():
        out['perf_report'][k] = v
    
    out['time'] = time_report

    df = pd.DataFrame(reader.data)

    return out, df

def prepare_sim_folder(base_folder_dir, subfolders):
    ret = {}
    # create base folder
    os.makedirs(base_folder_dir, exist_ok=True)
    # create subfolders
    for folder in subfolders:
        folder_dir = os.path.join(base_folder_dir, folder)
        ret[folder] = folder_dir
        os.makedirs(folder_dir, exist_ok=True)
    return ret

def np_to_json_serialize(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

def plot_data(df, output_image_path):
    # Create the figure and axis objects
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Accuracy on the left y-axis
    color1 = "blue"
    ax1.plot(df["br_inst_cnt"], df["accuracy"], color=color1, label="Accuracy")
    ax1.set_xlabel("Branch Instruction Count")
    ax1.set_ylabel("Accuracy", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Create a second y-axis for MPKI
    ax2 = ax1.twinx()
    color2 = "red"
    ax2.plot(df["br_inst_cnt"], df["mpki"], color=color2, label="MPKI")
    ax2.set_ylabel("MPKI", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add a title
    plt.title("Branch Prediction Accuracy and MPKI")

    # Save the plot as an image
    #output_image_path = "simulation_dual_axis_plot.png"  # Replace with your desired file name
    plt.savefig(output_image_path, dpi=300)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Tagebuilder!")
    parser.add_argument("-spec", type=str, help="spec name")
    #parser.add_argument("-o", "--optimized", action="store_true", help="Use optimzed tage sim")
    
    args = parser.parse_args()
    spec = args.spec
    #optimized = args.optimized
    print(args)

    # Get the current time
    current_time = datetime.now()
    # Format the time as a string suitable for file names
    file_name_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    sim_report_root = '/home/wonjongbot/tageBuilder/reports/'+f'sim_run_{file_name_time}'
    sim_list = [
        'SHORT_MOBILE-1',
        # 'SHORT_MOBILE-2',
        # 'SHORT_MOBILE-3',
        # 'SHORT_MOBILE-4',
        # 'LONG_MOBILE-1',
        # 'LONG_MOBILE-2',
        # 'LONG_MOBILE-3',
        # 'LONG_MOBILE-4',
        # 'SHORT_SERVER-1',
        # 'SHORT_SERVER-2',
        # 'SHORT_SERVER-3',
        # 'SHORT_SERVER-4',
        # 'LONG_SERVER-1',
        # 'LONG_SERVER-2',
        # 'LONG_SERVER-3',
        # 'LONG_SERVER-4',
    ]
    subdirs = prepare_sim_folder(sim_report_root, sim_list)

    #if optimized:
    for sim_name in sim_list:
        filepath = os.path.join(subdirs[sim_name], f'SIM_RESULT_{spec}_{sim_name}.json')
        with open(filepath, 'w') as f:
            out, df = run_single_sim(spec, sim_name) 
            json.dump(out, f, indent = 4, default = np_to_json_serialize)
        df_path = os.path.join(subdirs[sim_name], f'SIM_DATA_{spec}_{sim_name}.csv')
        img_path = os.path.join(subdirs[sim_name], f'SIM_PLOT_{spec}_{sim_name}.png')
        df.to_csv(df_path, index = False)
        plot_data(df, img_path)
    #else:
    #    pass 

    # if not optimized:
    #     with open(f'{settings.REPORT_DIR}UNOPTIMIZED_{spec}_{file_name_time}.txt', 'w') as f:
    #         #profiler = cProfile.Profile()
    #         #profiler.enable()
    #         start_wall = time.time()
    #         start_resources = resource.getrusage(resource.RUSAGE_SELF)

    #         out = main(NUM_INSTR = -1, spec_name= settings.SPEC_DIR+spec+".json")
            
    #         end_wall = time.time()
    #         end_resources = resource.getrusage(resource.RUSAGE_SELF)

    #         real_time = end_wall - start_wall
    #         user_time = end_resources.ru_utime - start_resources.ru_utime
    #         sys_time  = end_resources.ru_stime - start_resources.ru_stime
    #         time_str = f'\nTIME\n'
    #         time_str += f'    real {real_time:.3f} s\n'
    #         time_str += f'    user {user_time:.3f} s\n'
    #         time_str += f'    sys  {sys_time:.3f} s\n'

    #         out += time_str
    #         #profiler.disable()

    #         f.write(out)
    # else:
    #     with open(f'{settings.REPORT_DIR}OPTIMIZED_{spec}_{file_name_time}.txt', 'w') as f:
    #         #profiler = cProfile.Profile()
    #         #profiler.enable()

    #         start_wall = time.time()
    #         start_resources = resource.getrusage(resource.RUSAGE_SELF)
            
    #         out = main_optimized_tage(NUM_INSTR = -1, spec_name= settings.SPEC_DIR+spec+".yaml")
            
    #         end_wall = time.time()
    #         end_resources = resource.getrusage(resource.RUSAGE_SELF)

    #         real_time = end_wall - start_wall
    #         user_time = end_resources.ru_utime - start_resources.ru_utime
    #         sys_time  = end_resources.ru_stime - start_resources.ru_stime
    #         time_str = f'\nTIME\n'
    #         time_str += f'    real {real_time:.3f} s\n'
    #         time_str += f'    user {user_time:.3f} s\n'
    #         time_str += f'    sys  {sys_time:.3f} s\n'

    #         out += time_str
    #         #profiler.disable()

    #         f.write(out)

    #with open(f"{settings.REPORT_DIR}profiled/OPTIMIZED_profile_results_{configname}_{file_name_time}.txt", "w") as f:
    #  stats = pstats.Stats(profiler, stream=f)
    #  stats.sort_stats("cumulative")  # Sort by cumulative time
    #  stats.print_stats()