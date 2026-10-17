"""Minimal, read-only inference runtime retained with the final model."""
import argparse
from datetime import date
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_FEATURES=['latitude','longitude','season_sin','season_cos','tmax_lag1','tmax_mean7','tmax_max7',
    'rhmin_lag1','rhmin_mean7','rhmin_min7','windmax_lag1','windmax_max3','rain_lag1','rain_sum7','rain_sum14','rain_sum30','dry_days']
ANOMALIES=['tmax_lag1','tmax_mean7','rhmin_lag1','windmax_lag1','rain_sum7','rain_sum30']


def infer(bundle,rows):
    state=bundle['history']
    if not len(rows) or rows.target_date.isna().any() or rows.target_date.min()<=pd.Timestamp(state['fitted_through']):
        raise ValueError('Use dates after the training cutoff')
    if not rows.cell_id.isin(state['cell_codes']).all():raise ValueError('Unknown geographic cell')
    if rows.duplicated(['target_date','cell_id']).any():raise ValueError('Duplicate cell/date inputs')
    if 'feature_window_end' in rows:
        if not (pd.to_datetime(rows.feature_window_end)<rows.target_date).all():raise ValueError('Features must precede the target date')
    x=rows[BASE_FEATURES].copy()
    months=rows.target_date.dt.month.to_numpy()
    keys=[f'{cell}:{month}' for cell,month in zip(rows.cell_id,months)]
    if not all(key in state['climate'] for key in keys):raise ValueError('Missing cell/month historical statistics')
    baseline=np.array([state['baseline']['rates'].get(key,state['baseline']['global_rate']) for key in keys])
    x['cell_code']=rows.cell_id.map(state['cell_codes']).astype(int)
    x['month']=months
    x['weekday']=rows.target_date.dt.dayofweek.to_numpy()
    x['historical_prior']=baseline
    for name in ANOMALIES:
        stats=np.array([state['climate'][key][name] for key in keys])
        x[name+'_anomaly']=((rows[name].to_numpy()-stats[:,0])/stats[:,1]).clip(-12,12)
    for name in bundle['features']:
        if name not in x:x[name]=rows[name]
    x=x[bundle['features']]
    if not np.isfinite(x.to_numpy()).all():raise ValueError('Required model inputs are missing')
    weather=bundle['model'].predict_proba(x)[:,1]
    return (1-bundle['weather_weight'])*baseline+bundle['weather_weight']*weather


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description='Historical wildfire research inference; no website or external API calls.')
    parser.add_argument('--date',required=True,help='Target date, YYYY-MM-DD; bundled replay covers 2022–2024.')
    parser.add_argument('--input',type=Path,default=root/'datasets/model_inputs.csv.gz',help='Precomputed feature CSV; accepts gzip.')
    parser.add_argument('--cell',help='Optional cell identifier, e.g. r2c2')
    args=parser.parse_args()
    try:date.fromisoformat(args.date)
    except ValueError:parser.error('Date must be a real YYYY-MM-DD calendar date.')
    # Only trusted locally retained joblib files should be loaded.
    bundle=joblib.load(root/'model/model.joblib')
    data=pd.read_csv(args.input,float_precision='round_trip',parse_dates=['target_date'])
    rows=data[data.target_date==pd.Timestamp(args.date)].copy()
    if rows.empty:parser.error('No input features available for this target date.')
    try:rows['score']=infer(bundle,rows)
    except (ValueError,KeyError) as exc:parser.error(str(exc))
    rows=rows.sort_values(['score','cell_id'],ascending=[False,True])
    rows['rank']=np.arange(1,len(rows)+1)
    if args.cell:
        rows=rows[rows.cell_id==args.cell]
        if rows.empty:parser.error('Unknown cell identifier.')
    print(json.dumps({'model_version':bundle['version'],'model':bundle['selected_name'],'target_date':args.date,
        'trained_history_through':bundle['history']['fitted_through'],
        'mode':'Historical reanalysis research; raw score is not an operational probability.',
        'predictions':[{'cell_id':r.cell_id,'score':float(r.score),'rank':int(r.rank)} for r in rows.itertuples()]},indent=2,allow_nan=False))


if __name__=='__main__':main()
