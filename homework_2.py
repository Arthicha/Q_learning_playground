#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iml.dungeon import Dungeon
from iml.agent import DungeonAgent
from iml.visualization import DungeonVisualization
from iml.dungeon_maker import make_treasure_room, make_underground_maze, make_underground_maze_with_meat


# **Group members:** [your names]

# ## Part 1: Gridworld Environment

# ### 1.1 A drunken knight in a treasure room (4 pts)
# > (a) Implement a stochastic policy where you select actions uniformly at random. (0.5 pts)

# In[3]:


def my_reward_11(state, action, next_state, dungeon):
    if next_state == dungeon.end:
        return 1.0
    return 0.0

def my_action_11(state, q_value):
	# <implement your action here>
	# return ... (int)

def my_Q_update_11(agent, state, action, reward, next_state, dungeon):
    return None # use None for now

treasure_room = make_treasure_room()
vis_11 = DungeonVisualization(treasure_room)

drunken_knight = DungeonAgent(n_states=treasure_room.n_states, 
                             choose_action=my_action_11,
                             reward_func=my_reward_11,
                             update_q_values=my_Q_update_11)

# Run the agent for 1 episode & collect states, actions, rewards
states, actions, rewards = drunken_knight.run_episode(treasure_room, nstep=10, gamma=1.0)

# Show the map, Q-values, and path.
vis_11.show(states, actions, drunken_knight.q_values)


# > (b) Observe how you move and draw a Markov decision process (MDP) diagram for all four non-wall states. (0.5 pts)
# 
# *(Insert your MDP diagram photo/figure here)*

# > (c) Compute the theoretical probability distribution over a 2-step horizon analytically (without writing code) when you use the stochastic policy you implemented. (1 pt)
# 

# > (d) If you implement Q-learning with an episode length of 1 (`nstep = 1`) under current setup, will you manage to perform better? Why or why not? Explain your reasoning. (1 pt)
# 

# > (e) For a dungeon of width $N$ and height $M$, what is the minimum number of steps required to reach the goal from the starting position? Why? (1 pt)
# 

# ---
# ### 1.2 A tricky knight in an underground maze (6 pts)
# > (a) Using your stochastic policy $\pi$, implement the Q-learning update with a discount factor ($\gamma = 1$) and a maximum episode length of 30 steps ($n_{\text{step}} = 30$). Run consecutive episodes without resetting the Q-table. Explain your observations on how the Q-values change relative to the goal and starting locations. (1 pt)

# In[10]:


def my_reward_12(state, action, next_state, dungeon):
    if next_state == dungeon.end:
        return 1.0
    return 0.0

def my_action_12(state, q_values):
    return np.random.choice(4) 

def my_Q_update_12(agent, state, action, reward, next_state, dungeon):
	# <implement your Q update here>
	# Hint: lecture 9, page 18 <20/29>
	# return ... (float)

underground_maze = make_underground_maze(p_slippery=0.0)
vis_12 = DungeonVisualization(underground_maze)

tricky_knight = DungeonAgent(n_states=underground_maze.n_states, 
                             choose_action=my_action_12,
                             reward_func=my_reward_12,
                             update_q_values=my_Q_update_12)

training_reward = []
# program your knight here
# compute episodic reward
# append "training_reward"

# plot "training_reward"
vis_12.plot_training_reward(training_reward)


# > (b) By varying the discount factor $\gamma$, do you observe any differences? Summarize how the discount rate affects the results and explain why it is necessary. (1 pt)
# 

# > (c) Implement an $\epsilon$-greedy policy for your agent by varying $\epsilon$. Plot the sum of discounted rewards per episode over 100 training runs and explain your observations. (1 pt)

# > (d) Will you reach the goal every time? Explain why or why not. What should we do when this happens? (0.5 pts)

# > (e) Recreate the maze using `make_underground_maze_with_meat()`, update the reward function, set $\epsilon = 0.9$, $n_{\text{step}} = 50$, and $\gamma = 0.95$, and run the agent for 100 to 200 training runs. Explain your observations regarding your strategy (Q-values) and behavior. (1 pt)

# In[39]:


def my_reward_12(state, action, next_state, dungeon):
    if next_state == dungeon.end:
        return 1.0
    if dungeon.tiles[next_state] == "+":
        return 0.5
    return 0.0

def my_action_12(state, q_values):
	# <implement your action here>
	# return ... (int)

def my_Q_update_12(agent, state, action, reward, next_state, dungeon):
	# <implement your Q update here>
	# Hint: lecture 9, page 18 <20/29>
	# return ... (float)

underground_maze = make_underground_maze_with_meat(p_slippery=0.0)
vis_12 = DungeonVisualization(underground_maze)

tricky_knight = DungeonAgent(n_states=underground_maze.n_states, 
                             choose_action=my_action_12,
                             reward_func=my_reward_12,
                             update_q_values=my_Q_update_12)

training_reward = []
# program your knight here
# compute episodic reward
# append "training_reward"

# plot "training_reward"
vis_12.plot_training_reward(training_reward)


# > (f) Compare the state values $V(s)$ when you go directly to the treasure (e), collects free meat as you observed (+), or do both. Explain your results mathematically. (1.5 pts)
# 

# > (g) How might you modify the reward function or environment design to prevent this *reward hacking*? Propose and explain a solution. (Extra, +0.5 pts)

# ## Part 2: Tic-Tac-Toe

# ### (a)
# > Using the language of state value functions, explain how you would normally
# > solve tic-tac-toe. What is the meaning of the value of a position? What
# > would you say is the value of the starting state?

# ### (b)
# > Implement an agent that learns tic-tac-toe through Q-learning. By whatever
# > means necessary, ensure that the trained agent wins at least $99\%$ of the
# > time in a trial of $100,000$ fair games against the random opponent.

# ### (c)
# > When playing against the random opponent, are some choices of starting moves
# > better than others?

# ## Bonus: Multi-Armed Bandits

# ### (a)
# > Implement Thompson sampling for this problem. Over $100$ independent trials
# > with $5000$ iterations each, report the empirical fractions of times that
# > the agent chose the special arm.
