"""Builds cloud/wl_round3_bundle.zip: the code and measurement tables the round-3 runs need (~6 MB, no photos).

Usage (from the repository root or anywhere):  python New_Models_Baselines/03_Benchmark_Suite/cloud/make_bundle.py
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(SUITE, '..', '..'))
OUT = os.path.join(HERE, 'wl_round3_bundle.zip')

FILES = ['full_pipeline/olid_full_features.csv', 'full_pipeline/cv_fold_assignments.csv']
DIRS = [('Application/backend/app', ('.py',)),
        ('New_Models_Baselines/03_Benchmark_Suite', ('.py', '.sh')),
        ('New_Models_Baselines/03_Benchmark_Suite/results', ('.csv',))]

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(os.path.join(ROOT, f), 'wl_round3/' + f)
    for d, exts in DIRS:
        for name in sorted(os.listdir(os.path.join(ROOT, d))):
            p = os.path.join(ROOT, d, name)
            if os.path.isfile(p) and name.endswith(exts):
                data = open(p, 'rb').read()
                if name.endswith('.sh'):
                    data = data.replace(b'\r\n', b'\n')   # bash on Linux
                z.writestr(f'wl_round3/{d}/{name}', data)
print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')
