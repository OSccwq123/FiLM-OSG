#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('csvs', nargs='+')
ap.add_argument('--out', default='summary_experiments.json')
args=ap.parse_args()
import numpy as np
rows=[]
for path in args.csvs:
    p=Path(path)
    if not p.exists():
        continue
    with p.open() as f:
        for r in csv.DictReader(f):
            r['_source']=str(p)
            rows.append(r)
metrics=[k for k in rows[0].keys() if k not in {'model','seed','tag','_source'}] if rows else []
out={}
for r in rows:
    key=(r.get('model',''), r.get('tag',''))
    out.setdefault(str(key), {'model':key[0], 'tag':key[1], 'seeds':[], 'metrics':{}})
    out[str(key)]['seeds'].append(int(r.get('seed',0)))
for key,item in out.items():
    subset=[r for r in rows if str((r.get('model',''), r.get('tag',''))) == key]
    for m in metrics:
        vals=[]
        for r in subset:
            try: vals.append(float(r[m]))
            except Exception: pass
        if vals:
            a=np.array(vals)
            item['metrics'][m]={'mean':float(a.mean()),'std':float(a.std()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max()),'values':[float(x) for x in a]}
Path(args.out).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
