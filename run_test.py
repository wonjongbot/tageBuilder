import numpy as np
import treader
import tagebuilder
import json


def main(NUM_INSTR = -1, spec_name = "tage_custom.json"):
    out = ''
    filelist = [
        ('DIST-FP-1', '/home/wonjongbot/rv32-OoO-SoC/third-party/wi12_proj1/src/DIST-FP-1'),
        ('DIST-FP-2', '/home/wonjongbot/rv32-OoO-SoC/third-party/wi12_proj1/src/DIST-FP-2'),
        ('DIST-INT-1', '/home/wonjongbot/rv32-OoO-SoC/third-party/wi12_proj1/src/DIST-INT-1'),
        ('DIST-INT-2', '/home/wonjongbot/rv32-OoO-SoC/third-party/wi12_proj1/src/DIST-INT-2'),
        ('DIST-MM-1', '/home/wonjongbot/rv32-OoO-SoC/third-party/wi12_proj1/src/DIST-MM-1'),
        ('DIST-MM-2', '/home/wonjongbot/rv32-OoO-SoC/third-party/wi12_proj1/src/DIST-MM-2'),
        ('DIST-SERV-1', '/home/wonjongbot/rv32-OoO-SoC/third-party/wi12_proj1/src/DIST-SERV-1'),
        ('DIST-SERV-2', '/home/wonjongbot/rv32-OoO-SoC/third-party/wi12_proj1/src/DIST-SERV-2')
    ]
    with open(spec_name, 'r') as f:
        spec = json.load(f)

    sum_acc = 0
    sum_mispKPI = 0
    num_tests = len(filelist)

    predictor = tagebuilder.TAGEPredictor()
    predictor.init_tables(spec)
    out += predictor.sizelog

    for bm in filelist:
        print(f'TESTING {bm[0]}')
        reader = treader.TraceReader(bm[1])
        instr_cnt = 0
        while True:
        #for i in range(100):
            result = reader.read_branch()
            if result == -1:
                break  # End of file or read error
            # Process the instr_addr and br_taken as needed
            predictor.branch_pc = reader.instr_addr
            pred = predictor.make_prediction()
            reader.update_stats(pred)
            predictor.train_predictor(reader.br_taken)
            instr_cnt += 1
            if instr_cnt % 10000 == 0:
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
    with open(f'/home/wonjongbot/rv32-OoO-SoC/scripts/tageBuilder/reports/TAGE_CUSTOM.txt', 'w') as f:
        out = main(NUM_INSTR = -1, spec_name="/home/wonjongbot/rv32-OoO-SoC/scripts/tageBuilder/configs/tage_l.json")
        f.write(out)