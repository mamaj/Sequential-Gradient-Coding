import base64
import re
from pathlib import Path
import pickle
import numpy as np


def parse_log_duration(log):
    log = base64.b64decode(log).decode('utf-8')
    
    pattern = 'Duration: ([0-9.]*) ms'
    duration = re.search(pattern, log).group(1)
    return float(duration)


def load_windows_exp(nworkers, ninvokes, size, batch, region,
                     folder, complete_response=False):
    
    exp_folder = Path(folder)
    fname = f"w{nworkers}-n{ninvokes}-s{size}-b{batch}-{region}"
    if batch is None:
        fname = f"w{nworkers}-n{ninvokes}-s{size}-{region}"
    fpath = (exp_folder / fname).with_suffix('.pkl')


    with open(fpath, 'rb') as f:
        rounds = pickle.load(f)
        
    if not complete_response:
        for r in rounds:
            for res in r['results']:
                del res['response']
    return rounds


def get_durations(rounds, runtime=False):
    dur = [] 
    for round in rounds:
        if runtime:
            dur.append([w['runtime']/1000 for w in round['results']])
        else:
            dur.append([w['finished'] - w['started'] for w in round['results']])
    return np.array(dur) # (rounds, worker)
