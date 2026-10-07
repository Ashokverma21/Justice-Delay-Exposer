"""make_lookup.py - builds lookup.json for the app (dropdown values, label->code maps, error bands).
python make_lookup.py --cases sample_cases.csv.gz --acts sample_acts.csv.gz --ddl D:\\ddl --pred results\\predictions_test.csv.gz --out lookup.json
"""
import argparse, json
import numpy as np, pandas as pd

STATE_NAMES = {13: 'Uttar Pradesh', 26: 'Delhi', 22: 'Punjab', 14: 'Haryana', 9: 'Rajasthan', 1: 'Maharashtra', 17: 'Gujarat',
               10: 'Tamil Nadu', 4: 'Kerala', 3: 'Karnataka', 16: 'West Bengal', 11: 'Odisha', 8: 'Bihar', 23: 'Madhya Pradesh', 6: 'Assam'}

ap = argparse.ArgumentParser()
ap.add_argument('--cases', required=True); ap.add_argument('--acts', required=True)
ap.add_argument('--ddl', required=True); ap.add_argument('--pred', required=True)
ap.add_argument('--out', default='lookup.json'); ap.add_argument('--n-types', type=int, default=300); ap.add_argument('--n-purposes', type=int, default=200)
a = ap.parse_args()
import os
cases = pd.read_csv(a.cases, usecols=['ddl_case_id', 'year', 'state_code', 'dist_code', 'court_no', 'judge_position', 'type_name', 'purpose_name'])
L = {'states': {}, 'year_min': int(cases.year.min()), 'year_max': int(cases.year.max())}

# --- state > district > court
for st, g in cases.groupby('state_code'):
    dists = []
    for d, gd in g.groupby('dist_code'):
        cts = []
        for c, gc in gd.groupby('court_no'):
            if len(gc) >= 30:
                cts.append({'court_no': int(c), 'n': int(len(gc)), 'pos': str(gc.judge_position.mode().iat[0])})
        if cts:
            dists.append({'dist_code': int(d), 'n': int(len(gd)), 'courts': sorted(cts, key=lambda x: -x['n'])})
    L['states'][str(int(st))] = {'name': STATE_NAMES.get(int(st), 'State %d' % st), 'n': int(len(g)),
                                 'districts': sorted(dists, key=lambda x: x['dist_code'])}

# --- label -> {year: code}
def label_map(keyfile, col, scol, series, top):
    k = pd.read_csv(os.path.join(a.ddl, keyfile)).dropna(subset=[col])
    k[scol] = k[scol].fillna('').astype(str).str.lower().str.strip()
    k = k[k[scol] != '']
    cnt = series
    # label usage in sample
    kk = k.merge(cnt.rename('ns').reset_index(), on=['year', col], how='left').fillna({'ns': 0})
    use = kk.groupby(scol).ns.sum().sort_values(ascending=False).head(top)
    out = {}
    for lab in use.index:
        sub = kk[kk[scol] == lab].sort_values(['year', 'count'], ascending=[True, False]).drop_duplicates('year')
        out[lab] = {'n': int(use[lab]), 'codes': {str(int(y)): int(c) for y, c in zip(sub.year, sub[col])}}
    return out
tser = cases.groupby(['year', 'type_name']).size()
pser = cases.dropna(subset=['purpose_name']).assign(purpose_name=lambda d: d.purpose_name.astype(int)).groupby(['year', 'purpose_name']).size()
L['types'] = label_map('type_name_key.csv', 'type_name', 'type_name_s', tser, a.n_types)
L['purposes'] = label_map('purpose_name_key.csv', 'purpose_name', 'purpose_name_s', pser, a.n_purposes)

# --- acts
acts = pd.read_csv(a.acts, dtype=str)
ak = pd.read_csv(os.path.join(a.ddl, 'act_key.csv'), dtype=str).dropna(subset=['act'])
ak = ak[ak.act.str.fullmatch(r'\d+')].drop_duplicates('act').set_index('act').act_s
top = acts.drop_duplicates('ddl_case_id').act.value_counts().head(80)
L['acts'] = [{'code': int(c), 'name': str(ak.get(c, 'act %s' % c)).strip(), 'n': int(n)} for c, n in top.items() if str(c).isdigit()]

# --- error bands (log scale) from held-out predictions
p = pd.read_csv(a.pred)
p['has_act'] = p.ddl_case_id.isin(set(acts.ddl_case_id))
y = np.log1p(p.duration_days)
bands = {}
for m in ['A0', 'A1', 'B0', 'B1']:
    r = y - np.log1p(p['pred_' + m])
    bands[m] = {'q10': float(r.quantile(.1)), 'q90': float(r.quantile(.9))}
r = (y - np.log1p(p.pred_A1))[~p.has_act]; bands['A1_noact'] = {'q10': float(r.quantile(.1)), 'q90': float(r.quantile(.9))}
r = (y - np.log1p(p.pred_B1))[~p.has_act]; bands['B1_noact'] = {'q10': float(r.quantile(.1)), 'q90': float(r.quantile(.9))}
L['bands'] = bands
json.dump(L, open(a.out, 'w'), separators=(',', ':'))
print('saved', a.out, os.path.getsize(a.out) // 1024, 'KB |', len(L['types']), 'types', len(L['purposes']), 'purposes', len(L['acts']), 'acts')
print(bands)
