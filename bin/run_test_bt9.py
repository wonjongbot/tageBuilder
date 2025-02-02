import yaml
from tagebuilder_core import bt9reader
from tagebuilder_core import tage_predictor
from tagebuilder_core import settings
from tagebuilder_core import tage_optimized
from tagebuilder_core import plot_gen
from numba import njit

import numpy as np
import json
import time
import resource
import logging
import argparse
import multiprocessing
import pandas as pd
from tqdm import tqdm

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
        reader.logger.info(progressout)
        

def setup_logger(sim_id, sim_output_dir):
    logger = logging.getLogger(f"simulation_{sim_id}")
    logger.setLevel(logging.INFO)

    # File handler for process-specific logs
    log_file = os.path.join(sim_output_dir, f"simulation_{sim_id}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(fh)

    return logger

def run_single_sim(spec_name = "tage_sc_l", test_name = "SHORT-MOBILE-1", sim_id = 0, prog_queue=None, logger = None):
    out = {}
    fileinfo = (test_name, os.path.join(settings.CBP16_TRACE_DIR, test_name + '.bt9.trace.gz'))

    out['sim_id'] = 0
    out['timestamp'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    out['config'] = {}
    out['config']['predictor'] = spec_name
    out['config']['spec_file_dir'] = os.path.join(settings.SPEC_DIR, spec_name + '.yaml')
    out['config']['trace'] = fileinfo[0]
    out['config']['trace_file_dir'] = fileinfo[1]
    
    with open(out['config']['spec_file_dir'], 'r') as f:
        spec = yaml.safe_load(f)

    predictor = tage_optimized.TAGEPredictor(spec, logger)

    out['storage_report'] = {}
    for k,v in predictor.storage_report.items():
        out['storage_report'][k] = v

    logger.info(f'TESTING {fileinfo[0]} :: {fileinfo[1]}')

    reader = bt9reader.BT9Reader(fileinfo[1], logger)
    reader.init_tables()
    reader.init_predictor_scoreboard(predictor.num_tables)

    debug = 0

    expected_confidence_cols = [f'conf_{i}' for i in range(-4,4)]
    #prog_queue.put((sim_id, "start"))
    start_wall = time.time()
    start_resources = resource.getrusage(resource.RUSAGE_SELF)
    while True:
        # read batch by default
        b_size = 1_000_000
        result = reader.read_branch_batch(b_size)
        if result == -1:
            logger.info('INCOMPLETE FILE DETECTED')
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

        # Per Address stats
        # mask correct and incorrect predictions
        isCorrects = results['pred'] == reader.br_infoArr['taken']
        results_true = (isCorrects == 1)
        results_false = ~results_true

        # grab index from masks
        #idx_true = np.flatnonzero(results_true)
        #idx_false = np.flatnonzero(results_false)

        # grab address using index
        addr_true = reader.br_infoArr['addr'][results_true]
        addr_false = reader.br_infoArr['addr'][results_false]

        # update per address scoreboard (vectorized)
        # logger.info(reader.addr_scoreboard_df.loc[addr_true, 'num_correct_preds'])
        unique_addr_true, true_counts = np.unique(addr_true, return_counts=True)
        reader.addr_scoreboard_df.loc[unique_addr_true, 'num_correct_preds'] += true_counts
        # logger.info(reader.addr_scoreboard_df.loc[addr_true, 'num_correct_preds'])
        unique_addr_false, false_counts = np.unique(addr_false, return_counts=True)
        reader.addr_scoreboard_df.loc[unique_addr_false, 'num_incorrect_preds'] += false_counts

        # calculate transitions
        addrs = reader.br_infoArr['addr']
        takens = reader.br_infoArr['taken']
        
        # sort by address, use mergesort(stable) to preserve ordering for same addrs
        order = np.argsort(addrs, kind='mergesort')
        addrs = addrs[order]
        takens = takens[order]

        # boundaries for each addrs
        group_starts = np.concatenate(([0], np.where(np.diff(addrs) != 0 )[0] + 1))
        group_ends = np.concatenate((group_starts[1:], [len(addrs)]))

        # count transitions for each address
        if len(takens) > 1:
            diffs = (np.diff(takens) != 0).astype(np.int64)
        else:
            diffs = np.array([], dtype=np.int64)
        
        # cumulative sums to count transitions per group
        cumsum_diffs = np.concatenate(([0], np.cumsum(diffs)))
        group_transitions = cumsum_diffs[group_ends - 1] - cumsum_diffs[group_starts]

        # calculate total number of transitions (including previous batch's last value)
        group_addrs = addrs[group_starts]
        prev_takens = reader.addr_scoreboard_df.loc[group_addrs, 'prev_taken'].to_numpy()

        first_transitions = (prev_takens != takens[group_starts]).astype(np.int8)
        total_transitions = group_transitions + first_transitions
        
        # update pandas dataframe
        last_takens = takens[group_ends - 1]
        reader.addr_scoreboard_df.loc[group_addrs, 'trans'] += total_transitions
        reader.addr_scoreboard_df.loc[group_addrs, 'prev_taken'] = last_takens
        
        # Per predictor stats TODO: do everything vectorized? is this possible?
        # for id in range(predictor.num_tables):
        #     result_matching_id_flag = (results['pid'] == id)
        #     idx_matching_id = np.flatnonzero(result_matching_id_flag)
        #     # filter out for predictor in this iteration
        #     idx_true_per_id = np.intersect1d(idx_matching_id, idx_true)
        #     idx_false_per_id = np.intersect1d(idx_matching_id, idx_false)
        #     reader.predictor_scoreboard_df.loc[id, 'num_correct_preds'] += len(idx_true_per_id)
        #     reader.predictor_scoreboard_df.loc[id, 'num_incorrect_preds'] += len(idx_false_per_id)

        # per predictor stats (holy vectorization!)
        # mispredictions and correct predictions
        unique_pid_true, pid_counts_true = np.unique(results['pid'][results_true], return_counts=True)
        unique_pid_false, pid_counts_false = np.unique(results['pid'][results_false], return_counts=True)

        reader.predictor_scoreboard_df.loc[unique_pid_true, 'num_correct_preds'] += pid_counts_true
        reader.predictor_scoreboard_df.loc[unique_pid_false, 'num_incorrect_preds'] += pid_counts_false
        
        # alternate / main predictions
        is_alts = (results['is_alt'] == 1)
        is_mains = ~is_alts

        unique_pid_alt, pid_counts_alt = np.unique(results['pid'][is_alts], return_counts=True)
        unique_pid_main, pid_counts_main = np.unique(results['pid'][is_mains], return_counts=True)

        reader.predictor_scoreboard_df.loc[unique_pid_alt, 'used_as_alt'] += pid_counts_alt
        reader.predictor_scoreboard_df.loc[unique_pid_main, 'used_as_main'] += pid_counts_main

        # confidence among predictions (use pandas crosstab for funzies)
        mispred_df = pd.DataFrame(results[results_false])
        confidence_crosstab = pd.crosstab(mispred_df['pid'], mispred_df['ctr'])
        confidence_crosstab.columns = [f'conf_{col}' for col in confidence_crosstab.columns]
        confidence_crosstab = confidence_crosstab.reindex(columns=expected_confidence_cols, fill_value=0)
        reader.predictor_scoreboard_df = reader.predictor_scoreboard_df.add(confidence_crosstab, fill_value=0)
        # logger.info(confidence_crosstab)


        debug += len(addr_false)

        # update global statistics
        reader.report['correct_predictions'] += len(addr_true)
        reader.report['incorrect_predictions'] += len(addr_false)
        reader.report ['current_branch_instruction_count'] += (len(isCorrects))
        reader.report['current_instruction_count'] += np.sum(reader.br_infoArr['inst_cnt'])

        # for i, r in enumerate(results):
        #     # reader.update_stats(bool(r)) # remove function call overhead
        #     if bool(r):
        #         reader.addr_scoreboard[reader.br_infoArr[i]['addr']]['num_correct_preds'] += 1
        #         # reader.report['correct_predictions'] += 1
        #     else:
        #         reader.addr_scoreboard[reader.br_infoArr[i]['addr']]['num_incorrect_preds'] += 1
        #         # reader.report['incorrect_predictions'] += 1
        #     # reader.report['current_branch_instruction_count'] += 1
        #     # reader.report['current_instruction_count'] += (1 + int(reader.br_infoArr[i]['inst_cnt']))
        
        prog_queue.put((sim_id, reader.report['current_branch_instruction_count']))
        statHeartBeat(reader)
        if result == 1:
            #reader.report['current_branch_instruction_count'] += 1
            reader.report['is_sim_over'] = True
            logger.info('SIM IS OVER')
            prog_queue.put((sim_id, "done"))
            break
    
    assert(reader.report['is_sim_over'])

    logger.info(debug)

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
    time_report['sim_throughput'] = reader.report['current_branch_instruction_count']/real_time

    #addr_scoreboard_sorted = dict(sorted(reader.addr_scoreboard.items(), key=lambda x: x[1]['num_incorrect_preds'], reverse=True))

    out['perf_report'] = {}
    
    for k,v in reader.report.items():
        out['perf_report'][k] = v
    
    # # find a better way to do this
    # top_n_offender = {}
    # i = 0
    # for k,v in addr_scoreboard_sorted.items():
    #     if i > 10:
    #         break
    #     top_n_offender[hex(k)] = v
    #     i += 1
    #logger.info(top_n_offender)
    #logger.info(reader.addr_scoreboard)
    reader.predictor_scoreboard_df = reader.predictor_scoreboard_df[reader.predictor_scoreboard_structure]
    logger.info(reader.predictor_scoreboard_df)
    # out['perf_report']['top_n_offender'] = top_n_offender    
    out['time'] = time_report

    df_overall_mpki = pd.DataFrame(reader.data)
    logger.info(reader.addr_scoreboard_df)
    #df_per_addr_stats = pd.DataFrame.from_dict(reader.addr_scoreboard, orient='index')
    #reader.addr_scoreboard_df.index.name = 'br_addr'

    return out, df_overall_mpki, reader.addr_scoreboard_df, reader.predictor_scoreboard_df

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

def run_sim_wrapper(sim_dir, sim_name, spec, sim_id, prog_queue):
    """
    wrapper function to write sim outputs and graphics
    """
    filepath = os.path.join(sim_dir, f'SIM_RESULT_{spec}_{sim_name}.json')

    logger = setup_logger(sim_id, sim_dir)
    logger.setLevel(logging.INFO)
    logger.info(f"Simulation {sim_id} started.")

    out, df_overall_mpki, df_per_br_info, df_predictor_scoreboard = run_single_sim(spec, sim_name, sim_id, prog_queue, logger)
    # get n most incorrect predictions
    df_top_n_offender = df_per_br_info.nlargest(20, 'num_incorrect_preds')
    out['perf_report']['top_n_offender'] = df_top_n_offender.to_dict(orient = 'index')

    with open(filepath, 'w') as f:
        json.dump(out, f, indent = 4, default = np_to_json_serialize)

    df_overall_path = os.path.join(sim_dir, f'OVERALL_DATA.csv')
    df_per_branch_path = os.path.join(sim_dir, f'PER_BRANCH_DATA.csv')
    df_predictor_path = os.path.join(sim_dir, f'PER_PREDICTOR_DATA.csv')

    img_path = os.path.join(sim_dir, f'PLOT_OVERALL_MPKI_ACCURCY.png')
    img_storage_path = os.path.join(sim_dir, f'PLOT_STORAGE.png')
    img_top_n_addr = os.path.join(sim_dir, f'PLOT_TOP_N_ADDR.png')
    img_top_n_sum = os.path.join(sim_dir, f'PLOT_TOP_N_SUM.png')
    img_per_class = os.path.join(sim_dir, f'PLOT_PER_CLASS_STAT.png')

    df_overall_mpki.to_csv(df_overall_path, index = False)
    df_per_br_info.to_csv(df_per_branch_path, index = True)
    df_predictor_scoreboard.to_csv(df_predictor_path,index = True)
    
    # plot results
    plot_gen.plot_mpki_accuracy(df_overall_mpki, img_path)
    plot_gen.plot_storage_bar(out['storage_report'], img_storage_path, logger)
    plot_gen.plot_top_n_addr(out['perf_report']['top_n_offender'], out['perf_report']['incorrect_predictions'], img_top_n_addr)
    plot_gen.plot_top_n_sum(out['perf_report']['top_n_offender'], out['perf_report']['incorrect_predictions'], img_top_n_sum)
    plot_gen.plot_per_class(df_per_br_info, img_per_class)

def cli_progbar(sim_metadatas, sim_list, prog_queue):
    prog_bars = {}
    overall_prog = tqdm(
        total=len(sim_metadatas), desc="Overall Progress", position=len(sim_list), leave=True 
    )

    for sim_id, metadata in enumerate(sim_metadatas):
        prog_bars[sim_id] = tqdm(
            total=int(metadata['branch_instruction_count']), desc=f"{sim_list[sim_id]}", position=sim_id, leave=True 
        )
    
    while True:
        msg = prog_queue.get()
        if msg == "all_done":
            break
        sim_id, progress = msg

        if progress == "done":
            #prog_bars[sim_id].close()
            prog_bars[sim_id].n = prog_bars[sim_id].total
            prog_bars[sim_id].refresh()  # Ensure it shows 100% since that's prettier
            overall_prog.update(1)
        # elif progress == "start":
        #     prog_bars[sim_id].start_t = time.time()
        else:
            prog_bars[sim_id].n = progress
            prog_bars[sim_id].refresh()

    overall_prog.close()

def main_parallel():
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

    sim_report_root = os.path.join(settings.REPORT_DIR, f'sim_run_{file_name_time}')
    sim_list = [
        'SHORT_MOBILE-1',
        'SHORT_MOBILE-2',
        'SHORT_MOBILE-3',
        'SHORT_MOBILE-4',
        'LONG_MOBILE-1',
        'LONG_MOBILE-2',
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
    
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    num_proc = os.cpu_count()
    print(f"NUM PROC {num_proc}")

    # Read trace metadata for total # of br instructions
    trace_metadatas = []
    for trace_name in sim_list:
        trace_dir = os.path.join(settings.CBP16_TRACE_DIR, trace_name + '.bt9.trace.gz')
        print(trace_dir)
        reader = bt9reader.BT9Reader(trace_dir, None)
        reader.read_metadata()
        trace_metadatas.append(reader.metadata)
    # print(trace_metadatas)

    with multiprocessing.Pool(processes=num_proc) as pool:

        progbar_proc = multiprocessing.Process(
            target=cli_progbar, args=(trace_metadatas, sim_list, progress_queue)
        )
        progbar_proc.start()

        pool.starmap(
            run_sim_wrapper, 
            [(subdirs[sim_name], sim_name, spec, sim_id, progress_queue) for sim_id, sim_name in enumerate(sim_list)]
            )
        
        progress_queue.put("all_done")
        progbar_proc.join()

if __name__ == "__main__":
    main_parallel()