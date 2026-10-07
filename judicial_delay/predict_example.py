"""predict_example.py - use the trained model on any DDL-format case file.
python predict_example.py --model model_filing_time.pkl.gz --cases sample_cases.csv.gz --acts sample_acts.csv.gz --n 20
"""
import argparse, gzip, pickle
import pandas as pd
from delay_model import DelayModel   # required for unpickling

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='model_filing_time.pkl.gz'); ap.add_argument('--cases', required=True)
ap.add_argument('--acts', default=None); ap.add_argument('--n', type=int, default=20)
a = ap.parse_args()
m = pickle.load(gzip.open(a.model, 'rb') if a.model.endswith('.gz') else open(a.model, 'rb'))
cases = pd.read_csv(a.cases, nrows=a.n)
acts = pd.read_csv(a.acts, dtype=str) if a.acts else None
out = cases[['ddl_case_id', 'state_code', 'date_of_filing']].copy()
out['predicted_days'] = m.predict(cases, acts=acts).round(0)
if 'date_of_decision' in cases:
    out['actual_days'] = (pd.to_datetime(cases.date_of_decision) - pd.to_datetime(cases.date_of_filing)).dt.days
print(out.to_string(index=False))
