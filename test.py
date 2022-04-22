import numpy as np
from multiplexed_sgc import Multiplexed_SGC

delays = np.random.normal(size=(4, 6))
delays += 10
delays = np.abs(delays)


model = Multiplexed_SGC(n=4,
                B=2,
                W=3, 
                lambd=2,
                rounds=6, 
                mu=.2,
                delays=delays)


model.run()
model.state