from tagebuilder_core import treader
from tagebuilder_core import tage_predictor
from tagebuilder_core import legacy_settings

import numpy as np
import json
import time
import logging

import cProfile
import pstats

from datetime import datetime

legacy_settings.READ_BATCH = True
# Get the current time
current_time = datetime.now()

# Format the time as a string suitable for file names
file_name_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

configname = legacy_settings.SPEC_NAME

# the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{legacy_settings.REPORT_DIR}/logger/logger_{configname}_{file_name_time}.log"),
    ],
)

def main(NUM_INSTR = -1, spec_name = "tage_custom.json"):
    mainlogger = logging.getLogger(f"{__name__}")
    out = ''
    current_time = time.time()
    last_progress_time = current_time
    filelist = [
        ('DIST-FP-1', legacy_settings.TRACE_DIR + 'DIST-FP-1'),
        #('DIST-FP-2', settings.TRACE_DIR + 'DIST-FP-2'),
        #('DIST-INT-1', settings.TRACE_DIR + 'DIST-INT-1'),
        #('DIST-INT-2', settings.TRACE_DIR + 'DIST-INT-2'),
        #('DIST-MM-1', settings.TRACE_DIR + 'DIST-MM-1'),
        #('DIST-MM-2', settings.TRACE_DIR + 'DIST-MM-2'),
        #('DIST-SERV-1', settings.TRACE_DIR + 'DIST-SERV-1'),
        #('DIST-SERV-2', settings.TRACE_DIR + 'DIST-SERV-2')
    ]

    mainlogger.info('Tested traces:\n'+'\n'.join([f"('{name}', {path})" for name, path in filelist]))
    
    with open(spec_name, 'r') as f:
        spec = json.load(f)

    sum_acc = 0
    sum_mispKPI = 0
    num_tests = len(filelist)

    predictor = tage_predictor.TAGEPredictor()
    predictor.init_tables(spec)
    out += predictor.sizelog

    for bm in filelist:
        print(f'TESTING {bm[0]}')
        reader = treader.TraceReader(bm[1])
        instr_cnt = 0
        while True:
            current_time = time.time()
            if legacy_settings.READ_BATCH:
                b_size = 1024
                result = reader.read_branch_batch(b_size)
                if result == -1:
                    break
                for e in reader.br_info_arr:
                    reader.instr_addr = e[0]
                    reader.br_taken = e[1]

                    predictor.branch_pc = e[0]
                    pred = predictor.make_prediction()
                    reader.update_stats(pred)
                    predictor.train_predictor(e[1])
                    instr_cnt += 1
                    if instr_cnt == NUM_INSTR:
                        break
                #print(instr_cnt)
            else:
                result = reader.read_branch()
                if result == -1:
                    break  # End of file or read error
                # Process the instr_addr and br_taken as needed
                predictor.branch_pc = reader.instr_addr
                pred = predictor.make_prediction()
                reader.update_stats(pred)
                predictor.train_predictor(reader.br_taken)
                instr_cnt += 1
            if current_time - last_progress_time >= 0.5:
                last_progress_time = current_time
                if NUM_INSTR == -1:
                    print(f"{100 * instr_cnt / reader.stat_num_br_est:.3f}% :: {instr_cnt}", end="\r")
                else:
                    print(f"{100 * instr_cnt / NUM_INSTR:.3f}% :: {instr_cnt}", end="\r")
            if instr_cnt == NUM_INSTR:
                reader.stat_num_instr = instr_cnt
                break
            
        # Print the report stats
        out += f'--------------------\n'
        out += f"REPORT FOR {bm[0]}\n"
        out += reader.report_stats()

        (acc, mispKPI) = reader.get_stats()
        sum_acc += acc
        sum_mispKPI += mispKPI

        print(f'REPORT FOR {bm[0]}\n{reader.report_stats()}')

        #print(out_f)

    avg_acc = sum_acc / num_tests
    avg_mispKPI = sum_mispKPI / num_tests

    out += f"Avg accuracy: {avg_acc:.3f}%\nAvg misp/KPI: {avg_mispKPI:.3f}\n"
    return out

if __name__ == "__main__":

    with open(f'{legacy_settings.REPORT_DIR}{configname}_{file_name_time}.txt', 'w') as f:
        profiler = cProfile.Profile()
        profiler.enable()
        
        out = main(NUM_INSTR = -1, spec_name= legacy_settings.SPEC_DIR+configname+".json")
        
        profiler.disable()

        f.write(out)

    with open(f"{legacy_settings.REPORT_DIR}profiled/profile_results_{configname}_{file_name_time}.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats("cumulative")  # Sort by cumulative time
        stats.print_stats()