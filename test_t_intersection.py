from visualizer.t_intesection import TIntersectionSimulation
import numpy as np
from scipy.stats import beta
from scipy.stats import binom
from mdp import MDP
import matplotlib.pyplot as plt

class beta_dist:
    def __init__(self, a = 1, b = 1):
        self.a = a
        self.b = b
        
    #Get the beta pdf
    def get_pdf(self):
        x = np.linspace(0, 1, 1000)
        fx = beta.pdf(x, self.a, self.b)
        dens_dict = {'x': x, 'fx': fx}
        return(dens_dict)
        
    #Update parameters:
    def update_beta_params(self, n, num_successes):
        self.old_a = self.a
        self.old_b = self.b
        self.a = self.a + num_successes
        self.b = self.b + n - num_successes
    
    def get_mean(self):
        return self.a/(self.a + self.b)

class QMDPTIntersectionVisualizer(TIntersectionSimulation):
    def __init__(self, mdp, Q):
        self.goal_dist = beta_dist(1, 1)
        super().__init__(mdp, Q)
    
    def reset(self):
        self.goal_dist = beta_dist(1, 1)
        post_mean = self.goal_dist.get_mean()
        self.belief = [1.0-post_mean, post_mean]
        self.true_state = np.random.choice(self.mdp.num_s)
        self.t = 0
        self.our_t = 0
    
    def update_belief(self, observation):
        observations = [observation]
        # update the distribution
        self.goal_dist.update_beta_params(len(observations), sum(observations))
        post_mean = self.goal_dist.get_mean()
        self.belief = [1.0-post_mean, post_mean]
    
    def get_next_action(self):
        return np.argmax(self.belief @ self.Q)

def QMDP(V_star, belief, mdp=None):
    if mdp is None:
        raise ValueError("MDP cannot be None")

    numa, nums, R, P = mdp.get_mdp()
    # Initialize Q (state-action)
    Q = np.zeros((nums, numa))
    # loop through all actions
    for i in range(nums):
        Q[i,:] = R[i, :] + np.sum(V_star[i] * P[:, i, :])

    return np.argmax(belief @ Q)

class TIntersection(MDP):
    def __init__(self):
        self.gam = 0.9
        self.goal = ["G1", "G2"]
        self.actions = ["forward", "stop"]
        super().__init__(states=[self.goal], actions=self.actions)

        self.populate_data()

    def populate_data(self):
        # use self.add_route(s, a, s') from MDP to add route to MDP
        # use self.add_reward(s, a, r) from MDP to add reward
        self.add_route(["G1"], "forward", ["G1"], p=0.8)
        self.add_route(["G1"], "forward", ["G2"], p=0.2)
        self.add_route(["G2"], "forward", ["G1"], p=0.1)
        self.add_route(["G2"], "forward", ["G2"], p=0.9)

        self.add_route(["G1"], "stop", ["G1"], p=0.5)
        self.add_route(["G1"], "stop", ["G2"], p=0.5)
        self.add_route(["G2"], "stop", ["G1"], p=0.05)
        self.add_route(["G2"], "stop", ["G2"], p=0.95)

        self.add_reward(["G2"], "forward", -100)
        self.add_reward(["G1"], "forward", 10)
        self.add_reward(["G2"], "stop", -1)
        self.add_reward(["G1"], "stop", -1)

def value_iteration(threshold = .001, mdp=None):
    if mdp is None:
        raise ValueError("MDP cannot be None")
    numa, nums, R, P = mdp.get_mdp()
    V_star = np.zeros(nums)
    pi_star = np.zeros(nums)
    delta = np.inf
    while delta >= threshold:
        delta = 0
        for s in range(nums):
            v = V_star[s]
            V_star[s] = max(mdp.gam * V_star @ P[:, s, :] + R[s, :])
            pi_star[s] = np.argmax(mdp.gam * V_star @ P[:, s, :] + R[s, :])
            delta = max(delta, abs(v - V_star[s]))
    return V_star, pi_star

t_intersection = TIntersection()
V_star, pi_star = value_iteration(mdp=t_intersection)

numa, nums, R, P = t_intersection.get_mdp()
Q = np.zeros((nums, numa))
for i in range(numa):
  Q[:,i] = R[:, i] + np.sum(V_star @ P[:, :, i])

t_intersection_simulation = QMDPTIntersectionVisualizer(t_intersection, Q)

true_state = 1

t_intersection_simulation.reset()
t_intersection_simulation.set_true_state(true_state)

for i in range(100):
    t_intersection_simulation.step()
    t_intersection_simulation.plot()
    plt.pause(0.001)
    plt.clf()