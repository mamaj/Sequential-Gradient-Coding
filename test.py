import numpy as np
from multiplexed_sgc import Multiplexed_SGC
from utils import load_windows_exp, get_durations


load_windows_exp(
    nworkers=100,
    ninvokes=100,
    size=1000,
    region='Tokyo',
    folder='../aws-lambda/exp_long_3',
)

delays = np.random.normal(size=(4, 6))
delays += 10
delays = np.abs(delays)


model = Multiplexed_SGC(n=4,
                B=2,
                W=3, 
                lambd=2,
                rounds=6, 
                mu=.1,
                delays=delays)


model.run()
print(model.durations)
model.state