import numpy as np
from multiplexed_sgc import Multiplexed_SGC


model = Multiplexed_SGC(n=4,
                B=2,
                W=3, 
                lambd=2,
                rounds=6, 
                mu=1,
                delays=np.ones((4, 6)))


model.run()
