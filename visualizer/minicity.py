import numpy as np
import matplotlib.pyplot as plt
import math
import os

class MinicityVisualizer():
  def __init__(self, fig_prog_folder=None):
    self.fig_prog_folder = fig_prog_folder
    self.state_pos = np.array([
      [0, 1],
      [-1, 0],
      [1, 0],
      [3, 0],
      [0, 3],
      [-3, 0],
      [0, -3]
    ])
    
    n = 20
    r = 1
    self.roundabout = np.array([[math.cos(2*np.pi/n*x)*r for x in range(0,n+1)], [math.sin(2*np.pi/n*x)*r for x in range(0,n+1)]]).T
    self.current_pos = None
    self.goal = None
    self.iteration=0
    plt.figure(0)
  
  def reset(self, current_pos=None, goal=None):
    if current_pos is not None:
      self.current_pos = current_pos
    
    if goal is not None:
      self.goal = goal
  
  def get_transition(self, new_pos):
    """
    Get the transition scatters between the current pos and the new pos
    """
    transition_dict = {
        "0-1": [self.roundabout[i] for i in range(6, 10)],
        "0-2": [self.roundabout[i] for i in range(4, 0,-1)],
        "0-4": [[0, 1 + (2/7 * i)] for i in range(1, 7)],
        "1-5": [[-1 - (2/7 * i), 0] for i in range(1, 7)],
        "1-2": [self.roundabout[i] for i in range(11, 20)],
        "2-3": [[1 + (2/7 * i), 0] for i in range(1, 7)],
        "3-4": [[3, 0.75], [3, 1.5], [3, 2.25], [3, 3], [2.25, 3], [1.5, 3], [0.75, 3]],
        "3-6": [[3, -0.75], [3, -1.5], [3, -2.25], [3, -3], [2.25, -3], [1.5, -3], [0.75, -3]],
        "4-5": [[-0.75, 3], [-1.5, 3], [-2.25, 3], [-3, 3], [-3, 2.25], [-3, 1.5], [-3, 0.75]],
        "5-6": [[-3, -0.75], [-3, -1.5], [-3, -2.25], [-3, -3], [-2.25, -3], [-1.5, -3], [-0.75, -3]]
    }

    if new_pos > self.current_pos:
      return transition_dict["{}-{}".format(self.current_pos, new_pos)]
    else:
      return np.flip(np.array(transition_dict["{}-{}".format(new_pos, self.current_pos)]), axis = 0)
    
  def update_pos(self, new_pos=None, dir=None):
    if new_pos is not None:
      if new_pos != self.current_pos:
        transition = self.get_transition(new_pos)
        for t in transition:
          self.plot(current_pos=t, dir=dir)
        self.current_pos = new_pos
      self.plot(dir=dir)
    else:
      raise ValueError("new_pos is None, no need to update pos")

  def plot(self, current_pos=None, dir=None):
    plt.gca().add_patch(plt.Rectangle([-3, -3], width=6, height=6, fc="none", ec="k", lw=10, alpha=0.2))
    plt.gca().add_patch(plt.Circle((0, 0), 1, fc="none", ec="k", lw=10, alpha=0.2))
    plt.plot(self.state_pos[(2, 3), 0], self.state_pos[(2, 3), 1], lw=10, c="k", alpha=0.2)
    plt.plot(self.state_pos[(0, 4), 0], self.state_pos[(0, 4), 1], lw=10, c="k", alpha=0.2)
    plt.plot(self.state_pos[(1, 5), 0], self.state_pos[(1, 5), 1], lw=10, c="k", alpha=0.2)
    plt.scatter(self.state_pos[:, 0], self.state_pos[:, 1], s=200, zorder=10, c="k")

    # if there is current_pos, use it to plot the transition
    # else plot the current index of state for current_pos
    color = "b"
    if dir is not None:
      if dir == "cw": 
        color="y"
      else:
        color="b"
    if current_pos is not None:
      plt.scatter(current_pos[0], current_pos[1], s=300, zorder=20, c=color)
    elif self.current_pos is not None:
      plt.scatter(self.state_pos[self.current_pos, 0], self.state_pos[self.current_pos, 1], s=300, zorder=20, c=color)
    
    if self.goal is not None:
      plt.scatter(self.state_pos[self.goal, 0], self.state_pos[self.goal, 1], s=300, zorder=10, c="g")
    
    plt.gca().set_xlim((-3.5, 3.5))
    plt.gca().set_ylim((-3.5, 3.5))
    plt.gcf().set_size_inches(8, 8)

    # plt.pause(0.001)
    if self.fig_prog_folder is not None:
      plt.savefig(os.path.join(self.fig_prog_folder, "{}.png".format(self.iteration)), dpi=100)
    else:
      plt.pause(0.001)
    plt.clf()
    # plt.cla()
    # plt.close()

    self.iteration += 1
