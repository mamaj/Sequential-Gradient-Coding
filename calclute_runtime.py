import math
from functools import cache
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from gradient_coding import GradientCoding
from multiplexed_sgc import MultiplexedSGC
from selective_repeat_sgc import SelectiveRepeatSGC
from utils import get_durations, load_windows_exp



# parameters
folder = '../aws-lambda/delayprofile'
size = 500
region_name = 'Tokyo'
ninvokes = 300
workers = 300

rounds = 250  # number of jobs to complete
# buffer rounds is the maximum delay
max_delay = ninvokes - rounds  # total number of rounds profiled - number of jobs to complete

mu = 0.2

base_comp_time = 1. #TODO how to find this?



# parameter combinations:
gc_params = list(GradientCoding.param_combinations(workers))
srsgc_params = list(SelectiveRepeatSGC.param_combinations(workers, rounds, max_delay))
msgc_params = list(MultiplexedSGC.param_combinations(workers, rounds, max_delay))

print(f'GC: {len(gc_params)}')
print(f'SR-SGC:  {len(srsgc_params)}')
print(f'M-SGC:  {len(msgc_params)}')


# load delay profile
run_results = load_windows_exp(
    nworkers=workers,
    ninvokes=ninvokes,
    size=size,
    # batch=s, # this can be s to simulate normalized load 
    batch=1,
    region=region_name,
    folder=folder,
)
base_delays = get_durations(run_results).T # (workers, rounds)



# find runtimes
model_name = 'GC'

models = {
    'GC': GradientCoding,
    'SRSGC': SelectiveRepeatSGC,
    'MSGC': MultiplexedSGC,
}

Model = models[model_name]

runtimes = {'model name': model_name, 'duration': []}

params_combinations = list(Model.param_combinations(workers, rounds, max_delay))

for params in tqdm(params_combinations):
    
    # delays = base_delays + (s+1) * 0.
    delays = base_delays
    
    model = Model(workers, *params, rounds, mu, delays)
    model.run()
    runtime = model.durations.sum() + (model.total_rounds * (model.load) * 0.) # TODO: what to put instead of 1.
    runtimes['duration'].append({
        params: runtime
    })


def find_runtimes(params):
    model = Model(workers, *params, rounds, mu, delays)
    model.run()
    return model.durations.sum()


with ProcessPoolExecutor() as executor:
    futures = [
        executor.submit(find_runtimes, params)
        for params in tqdm(params_combinations)
        ]
    durations = [f.result() for f in futures]

