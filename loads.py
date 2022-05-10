import math
from functools import cache
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from gradient_coding import GradientCoding
from multiplexed_sgc import MultiplexedSGC
from selective_repeat_sgc import SelectiveRepeatSGC
from utils import get_durations, load_windows_exp


def srsgc_combinations(n, s):
    gc_load = GradientCoding.normalized_load(n, s)
    for W in range(1, n+1):
        for x in range(1, (W-1) + 1):
            if (W - 1) % x == 0:
                B = int((W - 1) / x)
                for lambd in range(1, n+1):
                    s_srsgc = math.ceil(lambd / (x+1))
                    if SelectiveRepeatSGC.normalized_load(n, s_srsgc) <= gc_load:
                        yield B, W, lambd, s_srsgc
        
                    
def msgc_combinations(n, s):
    gc_load = GradientCoding.normalized_load(n, s)
    for W in range(1, n+1):
        for B in range(1, W):
            for lambd in range(0, n):
                if MultiplexedSGC.normalized_load(n, B, W, lambd) <= gc_load:
                    yield B, W, lambd
        

def find_durations(n, s, rounds, mu, folder):
        print(f'started {s}')
        
        dur = {'s': s}

        run_results = load_windows_exp(
            nworkers=20,
            ninvokes=200,
            size=1000,
            batch=s,
            region='Canada',
            folder=folder,
        )
        delays = get_durations(run_results).T # (workers, rounds)

        # Gradient Coding
        gc = GradientCoding(n, s, rounds, mu, delays)
        gc.run()
        dur['best_gc'] = {'duration': gc.durations.sum(), 
                          'params': {'s': s}}

        # Selective Repeat SGC
        dur['srsgc'] = []
        for B, W, lambd, _ in srsgc_combinations(n, s):
            model = SelectiveRepeatSGC(n, B, W, lambd, rounds, mu, delays)
            model.run()
            dur['srsgc'].append({
                'duration': model.durations.sum(),
                'params': {'W': W, 'B': B, 'lambd': lambd}
            })
        dur['best_srsgc'] = min(dur['srsgc'], key=lambda x: x['duration'])

        # Multiplexed  SGC
        dur['msgc'] = []
        for B, W, lambd in msgc_combinations(n, s):
            model = MultiplexedSGC(n, B, W, lambd, rounds, mu, delays)
            model.run()
            dur['msgc'].append({
                'duration': model.durations.sum(),
                'params': {'W': W, 'B': B, 'lambd': lambd}
            })
        dur['best_msgc'] = min(dur['msgc'], key=lambda x: x['duration'])
        
        return dur


def main():
    # params
    n = 20
    rounds = 100
    mu = 0.1
    folder = '../aws-lambda/batch_numpy'
    
    # find best parameter combination for each choice of s
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(find_durations, n, s, rounds, mu, folder)
            for s in range(1, n)
            ]
        
    durations = [f.result() for f in futures]
    
    # save results
    file_path = folder.split('/')[-1] + '.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(durations, f)


if __name__ == '__main__':
    main()
    