# Report
---
Similar as the 2nd project which is training agents with continuous action spaces, this project also train the agents with continuos spaces. I use the same algorithm DDPG (Deep Deterministic Policy Gradient) for this project. The main difference is that the agents in training do not share the same Actor instance in this project. I modified training code acordingly to solve multi-agent problem in the [Tennis](hhttps://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

In the Tennis environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play as longer as possible.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The environment is considered solved, when the agents get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

# Learning algorithm

The action vector in the Tennis environment lies in continue space, therefore Deep Deterministic Policy Gradient (DDPG) is a good choice. However, simply using the same one for Reacher enviroment from the 2nd project cannot solve the Tennis enviroment.

DDPG is a kind of Actor-critic method which can leverage the strengths of both policy-based and value-based methods. Actor-critic methods consist of two models:
- The Critic updates the value function parameters.
- The Actor updates the policy parameters in the direction suggested by the Critic.

Unlike the solution for Reacher environment which all the agents share same Critic and Actor model, both agents in Tennis environment initialzed with its own Actor model,  but they share one Critic model.

The implementation of ddpg_agent is contained in [`maddpg_agent.py`](maddpg_agent.py). 

### Hyper Parameters using in ddpg_agent.py

- BUFFER_SIZE (int): replay buffer size
  - `BUFFER_SIZE = int(1e6)`
- BATCH_SIZ (int): minibatch size
  - `BATCH_SIZE = 256`
- GAMMA (float): discount factor
  - `GAMMA = 0.99`
- TAU (float): for soft update of target parameters
  - `TAU = 10e-3`
- LR_ACTOR (float): learning rate of the actor
  - `LR_ACTOR = 1e-4`
- LR_CRITIC (float): learning rate of the critic
  - `LR_CRITIC = 3e-4`
- LEARN_EVERY (int): learning timestep interval
  - `LEARN_EVERY = 10`
- LEARN_NUM (int): number of learning passes
  - `LEARN_NUM = 20`

The buffer size and batch size is suitable for the machine with large memory such as 32GB which is used in my machine. 
GAMMA should be large value near 1 and TAU should be small value (TAU is larger in this project than that in the 2nd project). 
Learning rate of the Actor and Critic are set by trial and error.
LEARN_EVERY and LEARN_NUM are also set by trial and error, and also make the system learn faster. 

### Neural network model architecture
The [Actor and Critic networks](model.py) are defined for DDPG algorithm are exactly same as the 2nd project.

- Actor network
  - 2 fully connected layers, both layers are 512 units
  - Fully connected layers activate with relu
  - Output layer for action space activate with tanh

- Critic network
  - 2 fully connected layers, both layers are 512 units
  - Fully connected layers activate with leaky_relu


### Other Hyper Parameters of DDPG

- n_episodes (int): maximum number of training episodes
  - `n_episodes=2000` 
- max_t (int): maximum number of timesteps per episode
  - `max_t=1000`

the maximum number episodes is set to relatively larger numbers in order to avoid early stopping, and the maximum of timesteps larger than 1000 is meaningless because the environment will always return `done` when `t` reaches 1000. 


# Plot of Rewards
The score for each episode, after taking the maximum over both agents:
![scores Plot](scores.png)

```
Episode 100	Average Score: 0.06
Episode 200	Average Score: 0.21
Episode 293	Score: [ 0.80  0.69]	Average Score: 0.50 (7.2 secs)
Environment solved in 193 episodes!	Average Score: 0.50
```

# Ideas for Future Work

Thanks to the experierence from the 2nd project, it is faster to tune the hyper parameters in DDPG algorithm this time. But the algorithm is still unstable and sometimes not working.

Distributed Distributional Deterministic Policy Gradients (D4PG) algorithm and Proximal Policy Optimization (PPO) algorithm could be explored they should be more robust and faster.

Prioritized experience replay also can be implemented to increase the efficiency by sampling the replay buffer with prioritized experience.




