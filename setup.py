import gdown
import tarfile

# CBP env folder (from craymichael/CBP-16-Simulation @ github)
cbp_url = 'https://drive.google.com/drive/folders/1VAdmqdOEFLvnRKkQQidxvGJA_C6S2RWo?usp=drive_link'

gdown.download_folder(cbp_url, quiet=False, use_cookies=False)

# Extract evaluation traces
evaltraces = '/Users/wonjongbot/tageBuilder/CBP16 Data/evaluationTraces.Final.tar'
extract_path = '/Users/wonjongbot/tageBuilder/CBP16 Data/'

with tarfile.open(evaltraces, "r:") as tar:
    tar.extractall(path=extract_path)
