import numpy as np
from multiplexed_sgc import MultiplexedSGC
from gradient_coding import GradientCoding
from selective_repeat_sgc import SelectiveRepeatSGC
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


model = MultiplexedSGC(n=4,
                        B=2,
                        W=3,
                        lambd=2,
                        rounds=6,
                        mu=.1,
                        delays=delays)

model = GradientCoding(n=4,
                       s=2,
                       rounds=6,
                       mu=.1,
                       delays=delays)


model = SelectiveRepeatSGC(n=4,
                           B=2,
                           W=3,
                           lambd=2,
                           rounds=6,
                           mu=.1,
                           delays=delays)

model.run()
print(model.durations)
print(model.state)