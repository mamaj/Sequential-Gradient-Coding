import math
from functools import cache
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from tqdm.contrib.concurrent import F
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
# runtime = model.durations.sum() + (model.total_rounds * (model.load) * 0.) 



# parameter combinations:
# gc_params = list(GradientCoding.param_combinations(workers))
# srsgc_params = list(SelectiveRepeatSGC.param_combinations(workers, rounds, max_delay))
# msgc_params = list(MultiplexedSGC.param_combinations(workers, rounds, max_delay))

# print(f'GC: {len(gc_params)}')
# print(f'SR-SGC:  {len(srsgc_params)}')
# print(f'M-SGC:  {len(msgc_params)}')



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

def find_runtimes(params):
    model = Model(workers, *params, rounds, mu, base_delays)
    model.run()
    return {params: model.durations.sum()}


model_name = 'SRSGC'

models = {
    'GC': GradientCoding,
    'SRSGC': SelectiveRepeatSGC,
    'MSGC': MultiplexedSGC,
}

Model = models[model_name]
params_combinations = list(Model.param_combinations(workers, rounds, max_delay))


if __name__ == '__main__':
        
    # with ProcessPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(find_runtimes, params)
    #         for params in tqdm(params_combinations)
    #         ]
    #     durations = [f.result() for f in tqdm(futures)]
    
    durations = process_map(find_runtimes, params_combinations,
                            chunksize=len(params_combinations)//cpu_count())
    
    print('done.')
    
    runtimes = {'model name': model_name, 'durations': durations}

    # save results
    file_path = f'{folder.split("/")[-1]}_{region_name}_mu0-2_{model_name}.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(runtimes, f)

