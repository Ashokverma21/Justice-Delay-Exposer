"""
delay_model.py - Judicial Delay Exposer: feature pipeline + packaged model.

USAGE (after training, with model.pkl next to this file):

    import pickle, gzip, pandas as pd
    from delay_model import DelayModel            # needed so pickle can find the class
    m = pickle.load(gzip.open('model_filing_time.pkl.gz', 'rb'))   # or model_with_purpose.pkl.gz
    cases = pd.read_csv('sample_cases.csv.gz')     # same columns as the DDL sample
    acts  = pd.read_csv('sample_acts.csv.gz', dtype=str)   # optional (ddl_case_id, act, section, ...)
    days  = m.predict(cases, acts=acts)      # each pickle holds ONE mode:
                                             #   model_filing_time.pkl.gz  = uses only what is known at filing
                                             #   model_with_purpose.pkl.gz = also uses the hearing-purpose bucket

Required case columns: ddl_case_id, year, state_code, dist_code, court_no, judge_position,
female_defendant, female_petitioner, female_adv_def, female_adv_pet, type_name, purpose_name,
date_of_filing  (date_of_decision is NOT needed for prediction).
"""
import numpy as np
import pandas as pd

TYPE_BUCKETS = ['bail', 'motor_accident', 'matrimonial_family', 'execution', 'negotiable_instr',
                'traffic_petty', 'criminal', 'civil', 'other']
PURPOSE_BUCKETS = ['missing', 'lok_adalat', 'judgment_order', 'arguments', 'evidence', 'bail',
                   'appearance_service', 'framing_pleadings', 'hearing', 'misc', 'other']
DISP_BUCKETS = ['missing', 'lok_adalat', 'settlement', 'conviction_acquittal', 'dismissed_withdrawn',
                'transferred_procedural', 'other']
BAIL_CATS = {'bailable': 1, 'non-bailable': 2, 'not applicable to section': 3, 'depends': 4, 'unclear, need to check': 5}

PURPOSE_RULES = [
    ('missing', r'^$'), ('lok_adalat', r'lok'),
    ('judgment_order', r'judg|order|decree|verdict|award|dispos|pronounce'),
    ('arguments', r'argu|final hearing|hearing of petition'),
    ('evidence', r'evid|witness|exam|cross|rebut|defen[cs]e|p\.?w|d\.?w'),
    ('bail', r'bail'),
    ('appearance_service', r'appear|appere|appeare|summon|process|notice|service|sr/ad|call on|await|steps|warrant|nbw|surety|accus'),
    ('framing_pleadings', r'issue|charge|plea|written statement|w\.s|reply|admission|consider|show cause|cognizan|enquiry|inquiry|framing|objection'),
    ('hearing', r'hear|first|for '), ('misc', r'misc|not heard|no sitting|other|general|adjourn')]
TYPE_RULES = [
    ('bail', r'bail|\bba\b|anticipatory|\bbl\b'), ('motor_accident', r'mact|motor|mcop|\bmacp\b|accident|mvc'),
    ('matrimonial_family', r'hma|divorce|matrim|maintenance|\bmc\b|family|guardian|\bgw\b|hindu marriage|dv act|domestic'),
    ('execution', r'exe|\bex\b|\bep\b|darkhast|execution'), ('negotiable_instr', r'nact|n\.i|ni act|138|cheque|negotiable'),
    ('traffic_petty', r'traffic|challan|police act|excise|\bsummary\b|\bsum\b|summ|petty|\bchi\b|\btrf\b'),
    ('criminal', r'\bcc\b|c\.c|\bst\b|\bstc|crl|cri|g r|gr case|\bsc\b|s\.c\.c|sessions|criminal|complaint|police|\bcr\b|fir'),
    ('civil', r'\bos\b|o\.s|\bcs\b|rct|rcs|civil|title|suit|\bcoma\b|\bmca\b|\bmjc\b|rent|petition|appeal|\bca\b|\bcma\b|\brca\b|\bmisc')]
DISP_RULES = [
    ('missing', r'missing|^$'), ('lok_adalat', r'lok adalat'),
    ('settlement', r'compromis|settle|plea bargain|compound|plead guilty|confession'),
    ('conviction_acquittal', r'convict|acquit|judg|fine|probation|prison|decided|award|allowed|partly'),
    ('dismissed_withdrawn', r'dismiss|withdraw|reject|quash|abated|not press|untrace|absconded|258'),
    ('transferred_procedural', r'transfer|procedur|committed|converted|remand|sine die|stayed|p\.o\.')]


import re as _re
_SYN = [(r'indianpenalcode|ipcpolice|^ipc.*', 'ipc'), (r'codeofcivilprocedure|civilprocedurecode|^cpc.*', 'cpc'),
        (r'codeofcriminalprocedure|criminalprocedurecode|^crpc.*', 'crpc'), (r'negotiableinstruments?act|^niact.*', 'niact'),
        (r'motorvehicles?act.*', 'mvact')]


def norm_act(name):
    t = _re.sub(r'[^a-z]', '', str(name).lower())
    for pat, rep in _SYN:
        if _re.search(pat, t):
            return rep
    return t.replace('vehicles', 'vehicle')


def bucket(s, rules, default='other'):
    out = pd.Series(default, index=s.index, dtype=object)
    done = pd.Series(False, index=s.index)
    for name, pat in rules:
        m = s.str.contains(pat, regex=True) & ~done
        out[m] = name
        done |= m
    return out


# ------------------------------------------------------------------ lookup tables
def _key_table(path, col, scol, rules, default, buckets):
    k = pd.read_csv(path)[['year', col, scol]].dropna(subset=[col])
    k[scol] = k[scol].fillna('').astype(str).str.lower().str.strip()
    b = bucket(k[scol], rules, default)
    labels, lab_id = np.unique(k[scol].values.astype(str), return_inverse=True)
    key = k.year.astype('int64').values * 1_000_000 + k[col].astype('int64').values
    o = np.argsort(key, kind='stable')
    return dict(key=key[o], label_id=lab_id[o].astype('int32'),
                bucket=np.array([buckets.index(x) for x in b], dtype='int8')[o],
                default=buckets.index(default if default in buckets else 'other'))


def build_roster(judges_csv, gap_min=15):
    """court key (state, dist, court_no, judge_position) -> (first_day, last_day, n_judges, gaps[g,2] in epoch days)"""
    j = pd.read_csv(judges_csv)
    j['start'] = pd.to_datetime(j.start_date, format='%d-%m-%Y', errors='coerce')
    j['end'] = pd.to_datetime(j.end_date, format='%d-%m-%Y', errors='coerce')
    j = j.dropna(subset=['start'])
    j['end'] = j.end.fillna(pd.Timestamp('2018-12-31')).clip(upper=pd.Timestamp('2018-12-31'))
    j = j[j.end >= j.start]
    j['s'] = j.start.values.astype('datetime64[D]').astype('int64')
    j['e'] = j.end.values.astype('datetime64[D]').astype('int64')
    roster = {}
    for k, g in j.sort_values('s').groupby(['state_code', 'dist_code', 'court_no', 'judge_position']):
        s, e = g.s.values, g.e.values
        cur_e = e[0]; gaps = []
        for si, ei in zip(s[1:], e[1:]):
            if si <= cur_e + 1:
                cur_e = max(cur_e, ei)
            else:
                if si - cur_e - 1 >= gap_min:
                    gaps.append((cur_e + 1, si - 1))
                cur_e = ei
        roster[(int(k[0]), int(k[1]), int(k[2]), str(k[3]))] = (int(s[0]), int(e.max()), len(g),
                                                                np.array(gaps, dtype='int64').reshape(-1, 2))
    return roster


def build_tables(ddl_dir, acts=None, gap_min=15, pos_list=None, top_pos=12, cases=None):
    import os
    T = {}
    T['type'] = _key_table(os.path.join(ddl_dir, 'type_name_key.csv'), 'type_name', 'type_name_s', TYPE_RULES, 'other', TYPE_BUCKETS)
    T['purpose'] = _key_table(os.path.join(ddl_dir, 'purpose_name_key.csv'), 'purpose_name', 'purpose_name_s', PURPOSE_RULES, 'other', PURPOSE_BUCKETS)
    T['disp'] = _key_table(os.path.join(ddl_dir, 'disp_name_key.csv'), 'disp_name', 'disp_name_s', DISP_RULES, 'other', DISP_BUCKETS)
    T['roster'] = build_roster(os.path.join(ddl_dir, 'judges_clean.csv'), gap_min)
    T['pos_list'] = pos_list if pos_list is not None else cases.judge_position.value_counts().head(top_pos).index.tolist()
    T['sections'] = pd.Index(acts.section.dropna().unique()) if acts is not None else pd.Index([])
    ak = os.path.join(ddl_dir, 'act_key.csv')
    T['act_codes'] = np.array([], dtype='int64'); T['act_norm_id'] = np.array([], dtype='int64')
    if os.path.exists(ak):
        k = pd.read_csv(ak, dtype=str).dropna(subset=['act'])
        k = k[k.act.str.fullmatch(r'\d+')]
        code = k.act.astype('int64').values
        ids = pd.factorize(k.act_s.fillna('').map(norm_act))[0]
        o = np.argsort(code)
        T['act_codes'] = code[o]; T['act_norm_id'] = ids[o].astype('int64')
    return T


def _lookup(tab, year, code, what, missing):
    q = year.astype('int64') * 1_000_000 + np.nan_to_num(np.asarray(code, dtype='float64'), nan=-1).astype('int64')
    idx = np.clip(np.searchsorted(tab['key'], q), 0, len(tab['key']) - 1)
    ok = tab['key'][idx] == q
    return np.where(ok, tab[what][idx], missing)


# ------------------------------------------------------------------ features
def _days(s):
    return pd.to_datetime(s, errors='coerce').values.astype('datetime64[D]').astype('int64')   # NaT -> very negative


def vacancy_features(f, d, st, dist, court, pos, roster):
    n = len(f)
    o = {k: np.zeros(n) for k in ['vac_days', 'vac_first180_cal', 'vac_at_filing']}
    o.update({k: np.zeros(n, int) for k in ['court_known', 'n_judges_court', 'in_roster_window', 'win180']})
    kd = pd.DataFrame({'s': st, 'd': dist, 'c': court, 'p': pos})
    for k, ii in kd.groupby(['s', 'd', 'c', 'p']).indices.items():
        r = roster.get((int(k[0]), int(k[1]), int(k[2]), str(k[3])))
        if r is None:
            continue
        first, last, nj, gaps = r
        fi = f[ii]
        o['court_known'][ii] = 1; o['n_judges_court'][ii] = nj
        o['win180'][ii] = ((fi >= first) & (fi + 180 <= last)).astype(int)
        tot = np.zeros(len(ii)); t180 = np.zeros(len(ii)); atf = np.zeros(len(ii))
        if d is not None:
            di = d[ii]
            o['in_roster_window'][ii] = ((fi >= first) & (di <= last)).astype(int)
        for gs, ge in gaps:
            t180 += np.clip(np.minimum(fi + 180, ge) - np.maximum(fi, gs) + 1, 0, None)
            atf += (fi >= gs) & (fi <= ge)
            if d is not None:
                tot += np.clip(np.minimum(di, ge) - np.maximum(fi, gs) + 1, 0, None)
        o['vac_first180_cal'][ii] = t180; o['vac_at_filing'][ii] = (atf > 0); o['vac_days'][ii] = tot
    return o


def make_features(df, acts, T):
    """-> S (static feature DataFrame), K (dict of int64 TE keys), X (extra columns for analysis, not model inputs)"""
    st = df.state_code.to_numpy('int64'); dist = df.dist_code.to_numpy('int64'); court = df.court_no.to_numpy('int64')
    year = df.year.to_numpy('int64')
    fd = pd.to_datetime(df.date_of_filing, errors='coerce')
    f = fd.values.astype('datetime64[D]').astype('int64')
    d = _days(df.date_of_decision) if 'date_of_decision' in df else None
    pos = df.judge_position.where(df.judge_position.isin(T['pos_list']), 'other_pos').astype(str)
    S = pd.DataFrame(index=df.index)
    S['state_code'] = st; S['dist_code'] = dist; S['court_no'] = court
    S['filing_year'] = fd.dt.year.values; S['filing_month'] = fd.dt.month.values; S['filing_dow'] = fd.dt.dayofweek.values
    sign = lambda s: s.astype(str).str.startswith('1').astype(int) - s.astype(str).str.startswith('-').astype(int)
    S['f_def'] = sign(df.female_defendant).values; S['f_pet'] = sign(df.female_petitioner).values
    S['female_adv_def'] = df.female_adv_def.values; S['female_adv_pet'] = df.female_adv_pet.values
    v = vacancy_features(f, d, st, dist, court, df.judge_position.astype(str).values, T['roster'])
    for k in ['vac_first180_cal', 'vac_at_filing', 'court_known', 'n_judges_court']:
        S[k] = v[k]
    S['vac180_flag'] = (v['vac_first180_cal'] > 0).astype(int)
    tb = _lookup(T['type'], year, df.type_name.values, 'bucket', T['type']['default'])
    S['type_bucket'] = tb
    S['pos'] = pd.Categorical(pos, categories=T['pos_list'] + ['other_pos']).codes
    pb = _lookup(T['purpose'], year, df.purpose_name.values, 'bucket', 0)
    S['purpose_bucket'] = pb
    # acts
    if acts is not None:
        a = acts.drop_duplicates('ddl_case_id').set_index('ddl_case_id').reindex(df.ddl_case_id.values)
        act = pd.to_numeric(a.act, errors='coerce').to_numpy()
        sec = T['sections'].get_indexer(a.section.fillna('__none__').values) if len(T['sections']) else np.full(len(df), -1)
        bail = a.bailable_ipc.map(BAIL_CATS).fillna(0).to_numpy()
        S['num_sections'] = pd.to_numeric(a.number_sections_ipc, errors='coerce').to_numpy()
        S['criminal_act'] = pd.to_numeric(a.criminal, errors='coerce').to_numpy()
    else:
        act = np.full(len(df), np.nan); sec = np.full(len(df), -1); bail = np.zeros(len(df))
        S['num_sections'] = np.nan; S['criminal_act'] = np.nan
    has = ~np.isnan(act)
    act_i = np.where(has, act, -1).astype('int64')
    if len(T.get('act_codes', [])):
        ix = np.clip(np.searchsorted(T['act_codes'], act_i), 0, len(T['act_codes']) - 1)
        actn = np.where(T['act_codes'][ix] == act_i, T['act_norm_id'][ix], -1).astype('int64')
    else:
        actn = np.full(len(df), -1, dtype='int64')
    S['has_act'] = has.astype(int); S['bailable'] = bail.astype(int)
    S['first_list_lag'] = (_days(df.date_first_list) - f) if 'date_first_list' in df else np.nan   # partial leakage: analysis only
    # TE keys
    dist_id = st * 1000 + dist
    court_id = dist_id * 100_000 + court
    type_lab = _lookup(T['type'], year, df.type_name.values, 'label_id', -1).astype('int64')
    purp_lab = _lookup(T['purpose'], year, df.purpose_name.values, 'label_id', -1).astype('int64')
    K = dict(court_id=court_id, dist_id=dist_id, court_type=court_id * 16 + tb, dist_type=dist_id * 16 + tb,
             type_raw=st * 1_000_000 + type_lab + 1, dist_year=dist_id * 10_000 + S['filing_year'].values,
             purp_raw=st * 1_000_000 + purp_lab + 1,
             act=act_i + 1, state_act=st * 1_000_000 + act_i + 1, dist_act=dist_id * 1_000_000 + act_i + 1,
             act_sec=(act_i + 1) * 100_000 + (sec.astype('int64') + 1),
             court_year=court_id * 10_000 + S['filing_year'].values,
             dist_type_year=(dist_id * 16 + tb) * 10_000 + S['filing_year'].values,
             court_act=court_id * 1_000_000 + act_i + 1,
             actn=actn + 1, actn_sec=(actn + 1) * 100_000 + (sec.astype('int64') + 1),
             state_actn=st * 1_000_000 + actn + 1, dist_actn=dist_id * 1_000_000 + actn + 1, court_actn=court_id * 1_000_000 + actn + 1)
    X = pd.DataFrame({'vac_days': v['vac_days'], 'in_roster_window': v['in_roster_window'], 'win180': v['win180']}, index=df.index)
    return S, K, X


# ------------------------------------------------------------------ target encoding
class TEncoder:
    def __init__(self, m=15):
        self.m = m

    def fit(self, keys, y):
        self.uk, inv = np.unique(keys, return_inverse=True)
        self.s = np.bincount(inv, weights=y, minlength=len(self.uk)); self.c = np.bincount(inv, minlength=len(self.uk)).astype('float64')
        self.gm = float(np.mean(y))
        return self

    def transform(self, keys):
        idx = np.clip(np.searchsorted(self.uk, keys), 0, len(self.uk) - 1)
        ok = self.uk[idx] == keys
        s = np.where(ok, self.s[idx], 0.0); c = np.where(ok, self.c[idx], 0.0)
        return (s + self.m * self.gm) / (c + self.m), c

    @staticmethod
    def oof(keys, y, fold_ids, m=15):
        """out-of-fold encoding for the training rows (each row never sees its own fold)"""
        uk, inv = np.unique(keys, return_inverse=True)
        S = np.bincount(inv, weights=y, minlength=len(uk)); C = np.bincount(inv, minlength=len(uk)).astype('float64')
        gm = float(np.mean(y)); te = np.zeros(len(keys)); n = np.zeros(len(keys))
        for fi in fold_ids:
            iv = inv[fi]
            Sf = np.bincount(iv, weights=y[fi], minlength=len(uk)); Cf = np.bincount(iv, minlength=len(uk)).astype('float64')
            s = S[iv] - Sf[iv]; c = C[iv] - Cf[iv]
            te[fi] = (s + m * gm) / (c + m); n[fi] = c
        return te, n


# ------------------------------------------------------------------ packaged model
class DelayModel:
    """holds tables, target encoders and one gradient-boosting model per mode"""
    def __init__(self):
        self.T = None; self.te = {}; self.models = {}; self.cols = {}; self.meta = {}

    def _matrix(self, S, K, mode):
        cols = self.cols[mode]; out = {}
        for c in cols:
            if c.startswith('te_'):
                out[c] = self.te[c[3:]].transform(K[c[3:]])[0]
            elif c.startswith('n_') and c[2:] in self.te:
                out[c] = self.te[c[2:]].transform(K[c[2:]])[1]
            else:
                out[c] = S[c].to_numpy()
        return np.column_stack([np.asarray(out[c], dtype='float32') for c in cols])

    def predict_log(self, df, acts=None, mode=None):
        mode = mode or next(iter(self.models))
        S, K, _ = make_features(df.reset_index(drop=True), acts, self.T)
        return self.models[mode].predict(self._matrix(S, K, mode))

    def predict(self, df, acts=None, mode=None):
        """predicted duration in days"""
        return np.expm1(self.predict_log(df, acts, mode))
