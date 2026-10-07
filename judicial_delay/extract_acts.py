"""
Act/section extractor: keeps ONLY the sampled cases from the big act-section file(s).

Run AFTER sample_states_final.py (it needs D:\\sample\\sample_cases.csv.gz):

    python extract_acts.py --acts "D:\\ddl\\acts_sections_*.csv"
or, for a single file:
    python extract_acts.py --acts "D:\\data\\my_acts_file.csv"

Output: D:\\sample\\sample_acts.csv.gz   (send me this with sample_cases.csv.gz)
Works in chunks, so the 2.5 GB file is never loaded fully.
"""
import argparse, glob, gzip, os, time
import pandas as pd

CHUNK = 500_000

ap = argparse.ArgumentParser()
ap.add_argument('--acts', required=True, help='path or glob of the act/section csv file(s)')
ap.add_argument('--sample', default=r'D:\sample\sample_cases.csv.gz')
ap.add_argument('--out', default=r'D:\sample\sample_acts.csv.gz')
a = ap.parse_args()

ids = set(pd.read_csv(a.sample, usecols=['ddl_case_id'], dtype=str).ddl_case_id)
print('sampled case ids:', len(ids))
files = sorted(glob.glob(a.acts))
if not files:
    raise SystemExit('No file matches: ' + a.acts)

t0 = time.time(); n_in = n_out = 0; first = True; matched = set()
with gzip.open(a.out, 'wt', encoding='utf-8', newline='') as fh:
    for f in files:
        head = pd.read_csv(f, nrows=0).columns.tolist()
        idcol = next((c for c in head if c == 'ddl_case_id'), None) or next((c for c in head if 'case_id' in c.lower()), None)
        if idcol is None:
            raise SystemExit('No case-id column in %s. Columns: %s' % (f, head))
        print('reading', os.path.basename(f), 'columns:', head, flush=True)
        for ch in pd.read_csv(f, chunksize=CHUNK, dtype=str, low_memory=False):
            n_in += len(ch)
            ch = ch[ch[idcol].isin(ids)]
            if ch.empty:
                continue
            ch = ch.rename(columns={idcol: 'ddl_case_id'})
            matched.update(ch.ddl_case_id)
            n_out += len(ch)
            ch.to_csv(fh, index=False, header=first)
            first = False
        print('  done %.0fs | rows read %d | kept %d' % (time.time() - t0, n_in, n_out), flush=True)

print('\nsampled cases with act/section info: %d of %d (%.1f%%)' % (len(matched), len(ids), 100 * len(matched) / len(ids)))
print('saved ->', a.out)
