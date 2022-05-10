import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

with open('batch_numpy.pkl', 'rb') as handle:
    durations = pickle.load(handle)


fig, ax = plt.subplots()


for method in ('gc', 'srsgc', 'msgc'):
    ax.plot(
        [dur['s'] for dur in durations],
        [dur[f'best_{method}']['duration'] for dur in durations],
        'o-',
        label=method)

ax.legend()
ax.set_xlabel('s')
ax.set_ylabel('run time (s)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid()