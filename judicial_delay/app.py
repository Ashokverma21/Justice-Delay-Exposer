"""
Judicial Delay Exposer - local web UI (Flask)

Run:      python app.py          then open  http://127.0.0.1:5000
Needs:    delay_model.py, lookup.json, models/model_filing_time.pkl.gz (+ optional models/model_with_purpose.pkl.gz),
          optional results.json (for the Findings tab)
"""
import gzip, io, json, os, pickle, sys
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, Response

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from delay_model import DelayModel   # noqa: F401  (needed to unpickle)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 400 * 1024 * 1024


def _find(*names):
    for n in names:
        for base in (HERE, os.path.join(HERE, 'models'), os.path.join(HERE, 'results')):
            p = os.path.join(base, n)
            if os.path.exists(p):
                return p
    return None


LOOK = json.load(open(_find('lookup.json'), encoding='utf-8'))
_rp = _find('results.json')
RES = json.load(open(_rp, encoding='utf-8')) if _rp else {}
MODELS = {}   # mode -> DelayModel that contains that mode
for fn in ['model_filing_time.pkl.gz', 'model_with_purpose.pkl.gz', 'model.pkl', 'model.pkl.gz']:
    p = _find(fn)
    if p:
        print('loading', p, flush=True)
        m = pickle.load(gzip.open(p, 'rb') if p.endswith('.gz') else open(p, 'rb'))
        for mode in m.models:
            MODELS.setdefault(mode, m)
if not MODELS:
    raise SystemExit('No model file found. Put model_filing_time.pkl.gz in the "models" folder next to app.py')
print('models ready:', list(MODELS), flush=True)

GENDER = {'unknown': '-9998 unclear', 'male': '0 male', 'female': '1 female'}


def nearest_code(codes, year):
    ys = sorted(int(y) for y in codes)
    y = min(ys, key=lambda v: abs(v - year))
    return codes[str(y)], y


def band(mode, has_act):
    b = LOOK['bands']
    key = ('A1' if mode == 'filing_time' else 'B1') + ('' if has_act else '_noact')
    return b[key]


def bucket_of(days):
    return 'Fast (< 3 months)' if days < 90 else ('Medium (3-12 months)' if days <= 365 else 'Slow (> 1 year)')


def predict_one(p):
    fdate = pd.to_datetime(p['filing_date'])
    year = int(np.clip(fdate.year, LOOK['year_min'], LOOK['year_max']))
    notes = []
    if fdate.year != year:
        notes.append('Filing year %d is outside the training range %d-%d; the model treated it as %d for lookups.'
                     % (fdate.year, LOOK['year_min'], LOOK['year_max'], year))
    st = int(p['state']); dist = int(p['district']); court = int(p['court'])
    pos = next((c['pos'] for d in LOOK['states'][str(st)]['districts'] if d['dist_code'] == dist
                for c in d['courts'] if c['court_no'] == court), 'other')
    tl = p['case_type']
    tcode, ty = nearest_code(LOOK['types'][tl]['codes'], year)
    if ty != year:
        notes.append('Case type "%s" not coded in %d; used the %d code.' % (tl, year, ty))
    pl = p.get('purpose') or ''
    mode = 'with_purpose' if (pl and 'with_purpose' in MODELS) else 'filing_time'
    row = dict(ddl_case_id='ui-0', year=year, state_code=st, dist_code=dist, court_no=court, judge_position=pos,
               female_defendant=GENDER.get(p.get('def_gender', 'unknown')), female_petitioner=GENDER.get(p.get('pet_gender', 'unknown')),
               female_adv_def=-9999, female_adv_pet=-9999, type_name=tcode,
               purpose_name=float(nearest_code(LOOK['purposes'][pl]['codes'], year)[0]) if pl else np.nan,
               date_of_filing=fdate.strftime('%Y-%m-%d'))
    df = pd.DataFrame([row])
    acts = None
    has_act = bool(p.get('act'))
    if has_act:
        acts = pd.DataFrame([dict(ddl_case_id='ui-0', act=str(int(p['act'])), section=None,
                                  bailable_ipc=p.get('bailable') or None,
                                  number_sections_ipc=p.get('num_sections') or None,
                                  criminal=p.get('criminal') if p.get('criminal') in ('0', '1') else None)])
    lg = float(MODELS[mode].predict_log(df, acts=acts, mode=mode)[0])
    b = band(mode, has_act)
    days = float(np.expm1(lg)); lo = float(np.expm1(lg + b['q10'])); hi = float(np.expm1(lg + b['q90']))
    st_name = LOOK['states'][str(st)]['name']
    return dict(days=round(days), lo=max(1, round(lo)), hi=round(hi), bucket=bucket_of(days), mode=mode, position=pos,
                state=st_name, state_median=RES.get('state_median_days', {}).get(st_name), notes=notes)


@app.get('/')
def index():
    return Response(PAGE, mimetype='text/html')


@app.get('/api/meta')
def meta():
    states = [{'code': int(k), 'name': v['name'], 'districts': [
        {'code': d['dist_code'], 'n': d['n'], 'courts': d['courts']} for d in v['districts']]}
        for k, v in sorted(LOOK['states'].items(), key=lambda kv: kv[1]['name'])]
    return jsonify(states=states,
                   types=[{'label': k, 'n': v['n']} for k, v in LOOK['types'].items()],
                   purposes=[{'label': k, 'n': v['n']} for k, v in LOOK['purposes'].items()],
                   acts=LOOK['acts'], year_min=LOOK['year_min'], year_max=LOOK['year_max'],
                   has_purpose_model='with_purpose' in MODELS)


@app.post('/api/predict')
def api_predict():
    try:
        return jsonify(ok=True, **predict_one(request.get_json()))
    except Exception as e:
        return jsonify(ok=False, error='%s: %s' % (type(e).__name__, e)), 400


@app.get('/api/findings')
def findings():
    return jsonify(RES)


def _read_upload(f, **kw):
    raw = f.read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), **kw)


@app.post('/api/batch')
def batch():
    try:
        cases = _read_upload(request.files['cases'])
        acts = _read_upload(request.files['acts'], dtype=str) if request.files.get('acts') and request.files['acts'].filename else None
        need = ['ddl_case_id', 'year', 'state_code', 'dist_code', 'court_no', 'judge_position', 'female_defendant', 'female_petitioner',
                'female_adv_def', 'female_adv_pet', 'type_name', 'date_of_filing']
        miss = [c for c in need if c not in cases.columns]
        if miss:
            return jsonify(ok=False, error='Missing columns in cases file: ' + ', '.join(miss)), 400
        mode = 'with_purpose' if ('purpose_name' in cases.columns and 'with_purpose' in MODELS) else 'filing_time'
        if 'purpose_name' not in cases.columns:
            cases['purpose_name'] = np.nan
        lg = MODELS[mode].predict_log(cases, acts=acts, mode=mode)
        out = cases[['ddl_case_id', 'state_code', 'date_of_filing']].copy()
        out['model_used'] = mode
        out['predicted_days'] = np.expm1(lg).round(0)
        out['low_10pct'] = np.expm1(lg + LOOK['bands']['A1' if mode == 'filing_time' else 'B1']['q10']).round(0)
        out['high_90pct'] = np.expm1(lg + LOOK['bands']['A1' if mode == 'filing_time' else 'B1']['q90']).round(0)
        out['speed_class'] = out.predicted_days.map(bucket_of)
        if 'date_of_decision' in cases.columns:
            out['actual_days'] = (pd.to_datetime(cases.date_of_decision, errors='coerce') - pd.to_datetime(cases.date_of_filing, errors='coerce')).dt.days
        buf = io.BytesIO(out.to_csv(index=False).encode('utf-8')); buf.seek(0)
        return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='predictions.csv')
    except Exception as e:
        return jsonify(ok=False, error='%s: %s' % (type(e).__name__, e)), 400


PAGE = open(os.path.join(HERE, 'page.html'), encoding='utf-8').read() if os.path.exists(os.path.join(HERE, 'page.html')) else '<h1>page.html missing</h1>'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print('\nOpen  http://127.0.0.1:%d  in your browser\n' % port)
    app.run(host='127.0.0.1', port=port, debug=False)
