"""
train_final.py - trains the Judicial Delay Exposer models and writes results + model.pkl

python train_final.py --cases sample_cases.csv.gz --acts sample_acts.csv.gz --ddl D:\\ddl --out results
(needs delay_model.py in the same folder; --ddl needs type/purpose/disp key csv + judges_clean.csv [+ act_key.csv optional])
"""
import argparse, json, os, pickle, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.inspection import permutation_importance
import delay_model as dm

ap = argparse.ArgumentParser()
ap.add_argument('--cases', required=True); ap.add_argument('--acts', default=None)
ap.add_argument('--ddl', required=True); ap.add_argument('--out', default='results')
ap.add_argument('--iters', type=int, default=4000); ap.add_argument('--leaf', type=int, default=127); ap.add_argument('--gap', type=int, default=15)
ap.add_argument('--limit', type=int, default=0, help='debug: use first N rows')
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
t0 = time.time()
log = lambda *a: print('[%4.0fs]' % (time.time() - t0), *a, flush=True)

df = pd.read_csv(A.cases, low_memory=False)
if A.limit: df = df.sample(A.limit, random_state=0)
df = df.reset_index(drop=True)
acts = pd.read_csv(A.acts, dtype=str, low_memory=False) if A.acts else None
fd = pd.to_datetime(df.date_of_filing, errors='coerce'); dd = pd.to_datetime(df.date_of_decision, errors='coerce')
df['duration_days'] = (dd - fd).dt.days
ok = df.duration_days.between(0, 3650) & fd.notna()
log('rows', len(df), '-> valid', int(ok.sum())); df = df[ok].reset_index(drop=True)
STATE = {1: 'Maharashtra', 3: 'Karnataka', 4: 'Kerala', 6: 'Assam', 8: 'Bihar', 9: 'Rajasthan', 10: 'Tamil Nadu', 11: 'Odisha',
         13: 'Uttar Pradesh', 14: 'Haryana', 16: 'West Bengal', 17: 'Gujarat', 22: 'Punjab', 23: 'Madhya Pradesh', 26: 'Delhi'}
df['state_name'] = df.state_code.map(STATE).fillna(df.state_code.astype(str))
assert df.dist_code.max() < 1000 and df.court_no.max() < 100_000, 'key packing overflow'

T = dm.build_tables(A.ddl, acts, A.gap, cases=df)
S, K, X = dm.make_features(df, acts, T)
log('features built', S.shape)
y = np.log1p(df.duration_days.values.astype('float64'))
dur = df.duration_days.values

# ---------------------------------------------------------- split + target encoding
rng = np.random.RandomState(42)
perm = rng.permutation(len(df)); ntest = int(0.2 * len(df))
te_idx, tr_idx = np.sort(perm[:ntest]), np.sort(perm[ntest:])
fold_ids = np.array_split(rng.permutation(len(tr_idx)), 5)           # positions within tr_idx
y_tr = y[tr_idx]

BASE_TE = ['court_id', 'dist_id', 'court_type', 'dist_type', 'type_raw', 'dist_year', 'court_year', 'dist_type_year']
ACT_TE = ['act', 'state_act', 'act_sec', 'dist_act', 'court_act']
PURP_TE = ['purp_raw']
BASE_NUM = ['state_code', 'dist_code', 'court_no', 'filing_year', 'filing_month', 'filing_dow', 'f_def', 'f_pet',
            'female_adv_def', 'female_adv_pet', 'vac_first180_cal', 'vac_at_filing', 'vac180_flag', 'court_known',
            'n_judges_court', 'type_bucket', 'pos']
ACT_NUM = ['num_sections', 'criminal_act', 'has_act', 'bailable']
CFG = {
    'A0 filing-time, no act':            (BASE_NUM, BASE_TE, 'filing_time'),
    'A1 filing-time + act/section':      (BASE_NUM + ACT_NUM, BASE_TE + ACT_TE, 'filing_time'),
    'B0 + purpose bucket, no act':       (BASE_NUM + ['purpose_bucket'], BASE_TE + PURP_TE, 'with_purpose'),
    'B1 + purpose bucket + act/section': (BASE_NUM + ACT_NUM + ['purpose_bucket'], BASE_TE + ACT_TE + PURP_TE, 'with_purpose'),
}
if acts is None:
    CFG = {k: v for k, v in CFG.items() if k[1] == '0'}
CATS = {'type_bucket', 'pos', 'purpose_bucket', 'bailable'}

enc = {}; te_tr = {}; te_te = {}
for name in sorted(set(sum([c[1] for c in CFG.values()], []))):
    te, n = dm.TEncoder.oof(K[name][tr_idx], y_tr, fold_ids)
    e = dm.TEncoder().fit(K[name][tr_idx], y_tr); enc[name] = e
    te_tr[name] = (te, n); te_te[name] = e.transform(K[name][te_idx])
log('target encodings done:', len(enc))


def build(cfg, which):
    num, tes, _ = cfg; cols = list(num); arr = []
    for c in num:
        arr.append(S[c].to_numpy()[tr_idx if which == 'tr' else te_idx])
    for t in tes:
        pair = te_tr[t] if which == 'tr' else te_te[t]
        arr += [pair[0], pair[1]]; cols += ['te_' + t, 'n_' + t]
    return np.column_stack([np.asarray(a, dtype='float32') for a in arr]), cols


def evaluate(p, idx):
    yt = y[idx]; d = dur[idx]; pd_ = np.expm1(p)
    b = lambda x: np.digitize(x, [90, 365])
    yb = d > 365
    return dict(r2_log=float(r2_score(yt, p)), r2_days=float(r2_score(d, pd_)), mae_days=float(mean_absolute_error(d, pd_)),
                bucket_acc_3class=float((b(d) == b(pd_)).mean()),
                over_1yr_acc=float(((pd_ > 365) == yb).mean()), over_1yr_baseline=float(max(yb.mean(), 1 - yb.mean())),
                over_1yr_auc=float(roc_auc_score(yb, pd_)))


results = {'n_rows': int(len(df)), 'n_train': int(len(tr_idx)), 'n_test': int(len(te_idx)), 'configs': {}}
model = dm.DelayModel(); model.T = T; model.te = enc
preds = {}; fitted = {}
for name, cfg in CFG.items():
    Xtr, cols = build(cfg, 'tr'); Xte, _ = build(cfg, 'te')
    mask = [c in CATS for c in cols]
    m = HistGradientBoostingRegressor(max_iter=A.iters, learning_rate=0.08, max_leaf_nodes=A.leaf, min_samples_leaf=40, l2_regularization=1.0,
                                      categorical_features=mask, early_stopping=True, validation_fraction=0.08, n_iter_no_change=30, random_state=0)
    m.fit(Xtr, y[tr_idx]); p = m.predict(Xte)
    r = evaluate(p, te_idx); r['iters'] = int(m.n_iter_); results['configs'][name] = r
    preds[name] = p; fitted[name] = (m, cols, Xte)
    log(name, {k: round(v, 3) for k, v in r.items()})
    json.dump(results, open(os.path.join(A.out, 'results.json'), 'w'), indent=1)
    del Xtr

# ---------------------------------------------------------- pack the model (one per mode: use the act-aware variant)
best = {'filing_time': 'A1 filing-time + act/section' if acts is not None else 'A0 filing-time, no act',
        'with_purpose': 'B1 + purpose bucket + act/section' if acts is not None else 'B0 + purpose bucket, no act'}
for mode, nm in best.items():
    m, cols, _ = fitted[nm]
    model.models[mode] = m; model.cols[mode] = cols; model.meta[mode] = dict(config=nm, **results['configs'][nm])
# make sure the packed encoders contain every key the packed models need
pickle.dump(model, open(os.path.join(A.out, 'model.pkl'), 'wb'), protocol=4)
log('model.pkl saved', round(os.path.getsize(os.path.join(A.out, 'model.pkl')) / 1e6, 1), 'MB')

# sanity: reload + predict a few raw rows must equal in-sample test predictions
chk = pickle.load(open(os.path.join(A.out, 'model.pkl'), 'rb'))
ii = te_idx[:2000]
pk = chk.predict_log(df.iloc[ii].drop(columns=['date_of_decision']), acts, mode='filing_time')
ref = preds[best['filing_time']][:2000]
results['pickle_check_max_abs_diff'] = float(np.abs(pk - ref).max())
log('pickle round-trip max |diff| (log scale):', results['pickle_check_max_abs_diff'])

# ---------------------------------------------------------- importance (act-aware filing-time model)
nm = best['filing_time']; m, cols, Xte = fitted[nm]
sub = np.random.RandomState(0).choice(len(Xte), 6000, replace=False)
pi = permutation_importance(m, Xte[sub], y[te_idx][sub], n_repeats=2, random_state=0, n_jobs=1)
imp = pd.Series(pi.importances_mean, index=cols)
grp = lambda c: c[3:] if c.startswith('te_') else (c[2:] if c.startswith('n_') and c[2:] in enc else c)
gi = imp.groupby(grp).sum().sort_values(ascending=False)
results['importance_filing_time_model'] = {k: round(float(v), 5) for k, v in gi.head(20).items()}
log('importance top:', list(gi.head(8).round(4).items()))

# ---------------------------------------------------------- state-wise + predictions file
te_df = df.iloc[te_idx][['ddl_case_id', 'state_name', 'duration_days']].copy()
for nm_, p in preds.items():
    te_df['pred_' + nm_.split()[0]] = np.expm1(p).round(1)
k = best['filing_time'].split()[0]
results['state_r2_filing_time'] = {s: round(float(r2_score(np.log1p(g.duration_days), np.log1p(g['pred_' + k]))), 3) for s, g in te_df.groupby('state_name')}
results['state_n_test'] = te_df.state_name.value_counts().to_dict()
te_df.to_csv(os.path.join(A.out, 'predictions_test.csv.gz'), index=False)

# ---------------------------------------------------------- descriptive findings (whole sample)
dstat = lambda g: dict(n=int(len(g)), mean=round(float(g.duration_days.mean()), 1), median=float(g.duration_days.median()))
df['vac_days'] = X.vac_days.values; df['in_win'] = X.in_roster_window.values; df['win180'] = X.win180.values
df['vac_first180_cal'] = S.vac_first180_cal.values; df['court_known'] = S.court_known.values
w = df[(df.in_win == 1)]; w2 = df[df.win180 == 1]
results['vacancy_whole_case'] = {('vacancy' if k else 'no_vacancy'): dstat(g) for k, g in w.groupby(w.vac_days > 0)}
results['vacancy_calendar180'] = {('vacancy' if k else 'no_vacancy'): dstat(g) for k, g in w2.groupby(w2.vac_first180_cal > 0)}
results['roster_match_pct'] = round(float(df.court_known.mean() * 100), 1)
year = df.year.values.astype('int64')
db = dm._lookup(T['disp'], year, df.disp_name.values, 'bucket', dm.DISP_BUCKETS.index('other'))
pbk = S.purpose_bucket.values
df['lok_disp'] = (db == dm.DISP_BUCKETS.index('lok_adalat')); df['lok_purp'] = (pbk == dm.PURPOSE_BUCKETS.index('lok_adalat'))
results['lok_disposal'] = {('lok' if k else 'other'): dstat(g) for k, g in df.groupby('lok_disp')}
results['lok_purpose'] = {('lok' if k else 'other'): dstat(g) for k, g in df.groupby('lok_purp')}
results['lok_by_state'] = {s: dict(n_lok=int(g.lok_disp.sum()), med_lok=float(g[g.lok_disp].duration_days.median()) if g.lok_disp.any() else None,
                                   med_other=float(g[~g.lok_disp].duration_days.median())) for s, g in df.groupby('state_name')}
results['state_median_days'] = {s: float(g.duration_days.median()) for s, g in df.groupby('state_name')}
results['state_n'] = df.state_name.value_counts().to_dict()
tb = S.type_bucket.values
results['type_bucket'] = {dm.TYPE_BUCKETS[i]: dstat(df[tb == i]) for i in range(len(dm.TYPE_BUCKETS)) if (tb == i).any()}
if acts is not None:
    df['has_act'] = S.has_act.values; df['criminal_act'] = S.criminal_act.values; df['bailable'] = S.bailable.values; df['act'] = K['act'] - 1
    results['act_coverage_pct'] = round(float(df.has_act.mean() * 100), 1)
    results['criminal_act'] = {str(k): dstat(g) for k, g in df[df.has_act == 1].groupby('criminal_act')}
    bn = {0: 'unknown', 1: 'bailable', 2: 'non-bailable', 3: 'n/a', 4: 'depends', 5: 'unclear'}
    results['bailable'] = {bn[k]: dstat(g) for k, g in df.groupby('bailable')}
    names = {}
    ak = os.path.join(A.ddl, 'act_key.csv')
    if os.path.exists(ak):
        akd = pd.read_csv(ak, dtype=str); names = dict(zip(akd.iloc[:, 3] if akd.shape[1] > 3 else akd.act, akd.act_s)) if 'act_s' in akd else {}
        names = dict(zip(akd['act'], akd['act_s'])) if {'act', 'act_s'} <= set(akd.columns) else {}
    top = df[df.has_act == 1].groupby('act').duration_days.agg(['count', 'mean', 'median']).sort_values('count', ascending=False).head(25)
    results['top_acts'] = [dict(act=int(a), name=str(names.get(str(a), names.get(a, '')))[:60], n=int(r['count']), mean=round(r['mean'], 1), median=float(r['median']))
                           for a, r in top.iterrows()]
json.dump(results, open(os.path.join(A.out, 'results.json'), 'w'), indent=1, default=float)
log('DONE')
