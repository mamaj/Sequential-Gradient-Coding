import numpy as np
import math

class SelectiveRepeatSGC:
    
    def __init__(self, n, B, W, lambd, rounds, mu, delays) -> None:
        # design parameters
        self.n = n # num workers
        self.B = B
        self.W = W
        self.lambd = lambd
        self.rounds = rounds
        self.mu = mu
        
        assert (W-1) % B == 0, 'B should devide W-1.'
        
        # parameters
        self.s = math.ceil((B * lambd) / (W - 1 + B))
        
        # delay profile
        self.delays = delays # (workers, rounds)
        
        # state of the master: (workers, round)
        self.state = np.full((n , rounds), np.nan) 
        self.durations = np.full((rounds, ), -1.)
                
    
    def run(self) -> None:
        for round_ in range(self.rounds):
            # perform round
            self.perform_round(round_)
            
            # decode
            job = self.get_decodable_job(round_)
            if job >= 0 and not self.is_decodable(job):
                raise RuntimeError(f'round {round_} is not decodable.')
                 
    
    def perform_round(self, round_) -> None:
        """ This will fill state(:,  round_) """
        
        # every worker by default returns task for round (round_)
        round_result = np.full((self.n, ), round_)
        
        # if there are v > s stragglers in round (round_ - B), v-s of
        # those stragglers repeat their (round_ - B) task instead of the 
        # current task
        decode_job = self.get_decodable_job(round_)
        if decode_job >= 0:
            prev_stragglers = np.flatnonzero(self.state[:, decode_job] == -1)
            if (v := prev_stragglers.size) > self.s:
                repeat_workers = prev_stragglers[0 : v - self.s]
                round_result[repeat_workers] = decode_job
            
        # apply stragglers
        delay = self.delays[:, round_]
        wait_time = delay.min() * (1 + self.mu) 
        is_straggler = delay > wait_time
        
        if self.follows_straggler_model(round_, is_straggler):
            # do not wait for all: apply straggler pattern
            round_result[is_straggler] = -1
            round_duration = wait_time
        else:
            # wait for all: do not apply stragglers
            round_duration = delay.max()
            
        # set round_result into state
        self.state[:, round_] = round_result
        self.durations[round_] = round_duration


    def is_decodable(self, job) -> bool:
        """
        checks whether a job can be decoded.
        To be able to decode job t, there should be at least n - s tasks received
        from workers in rounds  t and t + B.
        """
        task_results = self.task_results(job) # (2 * n, )
        return (task_results == job).sum() >= self.n - self.s
    
        
    def get_decodable_job(self, round_) -> int:
        """ returns the job decodable in round (round_) """
        return round_ - self.B
    
    
    def task_results(self, job) -> np.ndarray:
        """ returns the recieved tasks from workers in rounds job and job+B.
                shape: (2*n,) 
        """
        if job + self.B >= self.rounds:
            return self.state[:, job]
        else:
            return np.concatenate((self.state[:, job],
                                   self.state[:, job+self.B]))
    
    
    def follows_straggler_model(self, r, is_straggler) -> bool:
        """ Checks if at any given round, the spatial and temporal conditions 
            of (B, W, lambd)-bursty straggler model are met.
            
            1- spatial correlation: within the past W rounds, at most `lambd`
            unique stragglers.
            2- temporal correlation: if worker i is a straggler at the current
            round, it cannot be a straggler in [-W, -B] rounds relative to 
            the current round.
            
            r (int): current round idx.
            is_straggler (ndarray): boolean array of length n.
        """
        
        # 1. spatial cond: at most `lambd` unique stragglers over the 
        # past W rounds.
        state_window = self.state[:, np.maximum(0, r+1-self.W) : r]
        been_straggler = (state_window == -1).any(axis=1)
        num_stragglers = (been_straggler | is_straggler).sum()
        
        if num_stragglers > self.lambd:
            return False
        
        # 2. temporal cond: if worif worker i is a straggeler at the 
        # current round, it cannot be a straggeler in [-W, -B]:
        
        state_window = self.state[:, np.maximum(0, r+1-self.W) : np.maximum(0, r+1-self.B)]
        been_straggler = (state_window == -1).any(axis=1)
        
        if (been_straggler & is_straggler).any():
            return False
        
        return True
