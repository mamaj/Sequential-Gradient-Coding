import numpy as np


class Multiplexed_SGC:
    
    def __init__(self, n, B, W, lambd, rounds, mu, delays) -> None:
        # design parameters
        self.n = n
        self.B = B
        self.W = W
        self.lambd = lambd
        self.rounds = rounds
        self.mu = mu
        
        # delat profile
        self.delays = delays # (n, rounds)
        
        # parameters
        self.D1 = (W - 1)
        self.D2 = B
        self.minitasks = W - 1 + B
        self.T = W - 2 + B
        
        # state of the master: (worker, minitask, round)
        self.state = np.full((n, self.minitasks, rounds), -1) 
                
        # constants
        self.D1_TOKEN = 0
        self.D2_TOKENS = np.arange(B) + 1 # B tokens, one for each D2 group
    
    
    def run(self):
        for round_ in range(self.rounds):
            # perform round
            self.perform_round(round_)
            
            # decode
            t = round_ - self.T
            if t >= 0 and not self.is_decodable(t):
                raise RuntimeError(f'round {round_} is not decodable.')
                 
    
    def perform_round(self, round_) -> None:
        """ This will fill state(:, :, round_)
        """
        
        round_result = np.full((self.n, self.minitasks), -1) 
        
        # fill first D1 minitasks of workers with D1_TOKEN
        round_result[:, :self.D1] = self.D1_TOKEN
        
        # for next minitasks, if D1 of D1_TOKEN is present on diagonal, put 
        # corresponding D2_TOKEN, otherwise put D1_TOKEN
        for m in range(self.D2):
            t = round_ - self.D1 - m
            if t >= 0:
                num_d1 = (self.task_results(t) == self.D1_TOKEN).sum(axis=1)
                round_result[:, m + self.D1] = \
                    np.where(num_d1 > 1, self.D2_TOKENS[m], self.D1_TOKEN)
        
        # apply stragglers
        delay = self.delays[:, round_]
        wait_time = delay.min() * (1 + self.mu) 
        is_straggler = delay > wait_time
        round_result[is_straggler, :] = -1
        
        # set round_result into state
        self.state[:, :, round_] = round_result
    

    def is_decodable(self, t) -> bool:
        """
        To be able to decode:
            1. Each worker should have all of its D1 chunks.
            2. In total, at least `lambda` coded results from each of the
               B groups in D2.
            
        task_results: (n, minitasks) the diagonals of every worker: 
            minitasks = W-1 [=D1 slots] + B [=D2 slots]
        """
        
        task_results = self.task_results(t) # (n, minitasks) 
        
        # 1. Each worker should have D1 of D1_TOKEN
        num_d1 = (task_results == self.D1_TOKEN).sum(axis=1)
        if np.any(num_d1 < self.D1):
            return False
        
        # 2. There should be at least `lambd` of each D2_TONKENS in task_results
        num_d2 = (task_results.flatten()[:, None] == self.D2_TOKENS).sum(axis=-1)
        if np.any(num_d2 < self.lambd):
            return False
        
        return True
        
    
    def task_results(self, t):
        """ returns the diagonals of every worker for job t.
                shape: (n, minitasks) 
                minitasks = W-1 [=D1 slots] + B [=D2 slots]
        """
        
        # axis1 = minitask ax, axis2 = round ax
        return self.state.diagonal(axis1=1, axis2=2, offset=t)