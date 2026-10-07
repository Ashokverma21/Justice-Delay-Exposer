"""
Judicial Delay Exposer - FINAL sampler (streaming, low RAM)

Run (default: 15 states, 70,000 cases/state, D:\\ddl -> D:\\sample):
    python sample_states_final.py
Smaller/faster:   python sample_states_final.py --target 50000
Bigger:           python sample_states_final.py --target 100000

Output: D:\\sample\\sample_cases.csv.gz   (send me this one file)
        D:\\sample\\state_row_counts.csv
Pass 1 counts rows per state, pass 2 samples and writes straight to disk
(nothing big is kept in RAM). Invalid rows (bad dates, >10 years) are dropped on the fly.
"""
import argparse, glob, gzip, os, time
import numpy as np
import pandas as pd

CHUNK = 400_000
SEED = 42
STATES = [13, 26, 22, 14, 9, 1, 17, 10, 4, 3, 16, 11, 8, 23, 6]
# UP, Delhi, Punjab, Haryana, Rajasthan, Maharashtra, Gujarat, Tamil Nadu,
# Kerala, Karnataka, West Bengal, Odisha, Bihar, Madhya Pradesh, Assam
KEEP = ['ddl_case_id', 'year', 'state_code', 'dist_code', 'court_no', 'judge_position',
        'female_defendant', 'female_petitioner', 'female_adv_def', 'female_adv_pet',
        'type_name', 'purpose_name', 'disp_name',
        'date_of_filing', 'date_of_decision', 'date_first_list']   # no cino / last_list / next_list (not needed)
MARGIN = 1.25   # oversample a little: some rows are dropped as invalid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=r'D:\ddl')
    ap.add_argument('--out-dir', default=r'D:\sample')
    ap.add_argument('--target', type=int, default=70000, help='cases per state')
    ap.add_argument('--states', default=None, help='comma separated state codes (default: the 15 above)')
    a = ap.parse_args()
    states = [int(s) for s in a.states.split(',')] if a.states else STATES
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.data_dir, 'cases_20??.csv')))
    if not files:
        raise SystemExit('cases_20XX.csv files not found in ' + a.data_dir)
    t0 = time.time()

    # ---- pass 1: rows per state
    tot = {}
    for f in files:
        for ch in pd.read_csv(f, usecols=['state_code'], chunksize=CHUNK):
            for st, n in ch.state_code.value_counts().items():
                tot[st] = tot.get(st, 0) + int(n)
        print('counted', os.path.basename(f), '%.0fs' % (time.time() - t0), flush=True)
    pd.Series(tot).sort_index().to_csv(os.path.join(a.out_dir, 'state_row_counts.csv'), header=['rows'])
    prob = {s: min(1.0, a.target * MARGIN / tot[s]) for s in states if s in tot}
    missing = [s for s in states if s not in tot]
    if missing:
        print('WARNING: state codes not found in data:', missing)

    # ---- pass 2: sample + validate + stream to gzip
    rng = np.random.default_rng(SEED)
    out = os.path.join(a.out_dir, 'sample_cases.csv.gz')
    counts = {s: 0 for s in prob}
    first = True
    with gzip.open(out, 'wt', encoding='utf-8', newline='') as fh:
        for f in files:
            for ch in pd.read_csv(f, usecols=KEEP, chunksize=CHUNK, low_memory=False):
                ch = ch[ch.state_code.isin(prob.keys())]
                if ch.empty:
                    continue
                keep = rng.random(len(ch)) < ch.state_code.map(prob).to_numpy()
                ch = ch[keep]
                if ch.empty:
                    continue
                fi = pd.to_datetime(ch.date_of_filing, errors='coerce')
                de = pd.to_datetime(ch.date_of_decision, errors='coerce')
                dur = (de - fi).dt.days
                ok = fi.notna() & de.notna() & (fi >= '1990-01-01') & (de <= '2018-12-31') & (dur >= 0) & (dur <= 3650)
                ch = ch[ok.to_numpy()]
                for s, n in ch.state_code.value_counts().items():
                    counts[s] += int(n)
                ch.to_csv(fh, index=False, header=first)
                first = False
            print('sampled', os.path.basename(f), '%.0fs' % (time.time() - t0), flush=True)

    print('\nrows per state (valid):')
    for s in sorted(counts):
        flag = '' if counts[s] >= a.target * 0.9 else '   <-- below target'
        print('  state %2d: %7d%s' % (s, counts[s], flag))
    print('TOTAL', sum(counts.values()), '->', out, '| %.0f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
