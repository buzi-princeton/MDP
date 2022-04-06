from abc import ABC, abstractmethod
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

class TIntersectionSimulation(ABC):
  def __init__(self, mdp, Q):
    self.mdp = mdp
    self.true_state = None
    
    # Let the horizon be long enough so that we have enough time to gather observation
    self.T = 1
    self.N = 50
    self.t = 0
    self.our_t = 0
    self.dt = self.T / self.N

    self.belief = None
    self.Q = Q
    
    # create sample trajectories
    self.G1_traj_x = [3.85] * self.N
    self.G1_traj_y = np.flip(np.arange(24, 28, (28-24)/self.N))
    
    self.G2_traj_x = CubicSpline(np.arange(0, self.T, self.T/7), np.array([3.85, 4, 5, 7.5, 12.5, 17.5, 24]))
    self.G2_traj_y = CubicSpline(np.arange(0, self.T, self.T/7), np.array([28, 25, 23.5, 22.5, 21.85, 21.85, 21.85]))
    
    self.our_traj_x = CubicSpline(np.arange(0, self.T, self.T/7), np.array([10, 7.5, 5.7, 4.7, 4.1, 3.85, 3.85]))
    self.our_traj_y = CubicSpline(np.arange(0, self.T, self.T/7), np.array([25.55, 24.5, 23.7, 22.8, 21, 15, 10]))

    self.reset()

  @abstractmethod
  def reset(self):
    raise NotImplementedError("reset is not implemented")

  def plot_road(self, ax):
    road_rules = {
      "x_min": 2,
      "x_max": 9.4,
      "y_max": 27.4,
      "y_min": 20,
      "width": 3.7
    }

    x_max = 25
    y_max = 40

    x_center = road_rules["x_min"] + 0.5 * (road_rules["x_max"] - road_rules["x_min"])
    y_center = road_rules["y_min"] + 0.5 * (road_rules["y_max"] - road_rules["y_min"])

    ax.plot([road_rules["x_min"], road_rules["x_min"]], [0, y_max], c="k", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], road_rules["x_max"]], [0, road_rules["y_min"]], c="k", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], road_rules["x_max"]], [road_rules["y_max"], y_max], c="k", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_min"], road_rules["x_min"]], [road_rules["y_min"], road_rules["y_min"]], c="k", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], x_max], [road_rules["y_min"], road_rules["y_min"]], c="k", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_min"], road_rules["x_min"]], [road_rules["y_max"], road_rules["y_max"]], c="k", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], x_max], [road_rules["y_max"], road_rules["y_max"]], c="k", linewidth = 2, zorder = -1)
    ax.plot([x_center, x_center], [0, y_max], "--", c = 'k', linewidth = 5, dashes=(5, 5), zorder = -1)
    ax.plot([road_rules["x_max"], x_max], [y_center, y_center], "--", c = 'k', linewidth = 5, dashes=(5, 5), zorder = -1)

  @abstractmethod
  def update_belief(self, observation):
    raise NotImplementedError("update_belief is not implemented")

  @abstractmethod
  def get_next_action(self):
    raise NotImplementedError("get_next_action is not implemented")

  def step(self):
    if self.t < self.T - self.dt:
      self.t += self.dt
    action = self.get_next_action()
    if action == 0:
      if self.our_t < self.T - self.dt:
        self.our_t += self.dt
    self.update_belief(self.true_state)

  def set_true_state(self, true_state):
    self.true_state = true_state

  def plot(self):
    # plot sample T-intersection and three sample trajectories
    # (one for our car and 2 for the other car)
    plt.figure(0, figsize=(20/3, 15/3))
    plt.axis("off")

    plt.plot(self.G1_traj_x, self.G1_traj_y, "--k", alpha=0.5)
    plt.plot(self.G2_traj_x(np.linspace(0, self.T, self.N)), self.G2_traj_y(np.linspace(0, self.T, self.N)), "--r", alpha=0.5)
    plt.plot(self.our_traj_x(np.linspace(0, self.T, self.N)), self.our_traj_y(np.linspace(0, self.T, self.N)), "--b", alpha=0.5)

    # draw car footprint based on self.t
    if self.true_state == 0:
      plt.scatter(self.G1_traj_x[int(self.t/self.dt)], self.G1_traj_y[int(self.t/self.dt)], c="r", s=200)
    else:
      plt.scatter(self.G2_traj_x(self.t), self.G2_traj_y(self.t), c="r", s=200)

    plt.scatter(self.our_traj_x(self.our_t), self.our_traj_y(self.our_t), c="b", s=200)

    plt.xlim(0, 20)
    plt.ylim(15, 30)

    ax = plt.gca()
    self.plot_road(ax)