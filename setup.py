import gdown
import tarfile
import os

# get current working directory
cwd = os.getcwd()

# CBP env folder (from craymichael/CBP-16-Simulation @ github)
cbp_url = 'https://drive.google.com/drive/folders/1VAdmqdOEFLvnRKkQQidxvGJA_C6S2RWo?usp=drive_link'
gdown.download_folder(cbp_url, quiet=False, use_cookies=False)


# Extract evaluation traces
evaltraces_dir = os.path.join(cwd, 'CBP16 Data/evaluationTraces.Final.tar')
extracted_dir = os.path.join(cwd, 'CBP16 Data/')

with tarfile.open(evaltraces_dir, "r:") as tar:
    # tar.extractall(path=extracted_dir)
    print(f'Trace extracted to {extracted_dir}')


# setup directories used in sim
coredir = os.path.join(cwd, 'tagebuilder_core')


# setup settings to resolve directory dependencies
settingfile_dir = os.path.join(cwd, 'tagebuilder_core/settings.py')

settings_components = {
    'CBP16_TRACE_DIR': os.path.join(extracted_dir, 'evaluationTraces'),
    'REPORT_DIR': os.path.join(cwd, 'reports'),
    'SPEC_DIR': os.path.join(cwd, 'configs')
}

with open(settingfile_dir, 'w') as f:
    for k, v in settings_components.items():
        f.write(f'{k} = \'{v}\'\n')