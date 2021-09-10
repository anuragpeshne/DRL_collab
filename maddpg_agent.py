import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 8e-3              # for soft update of target parameters
LR_ACTOR = 3e-3         # learning rate of the actor
LR_CRITIC = 4e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    """Multi Agent DDPG agent"""

    def __init__(self, state_size, action_size, random_seed, num_agents):
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.action_size = action_size
        self.agents = [Agent(state_size, action_size, num_agents, random_seed) for i in range(num_agents)]

    def act(self, state):
        actions = [agent.act(observation) for agent, observation in zip(self.agents, state)]
        return actions

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        for agent_id, agent in enumerate(self.agents):
            states, actions, rewards, next_states, dones = experiences

            observations = states.view(BATCH_SIZE, -1)
            actions = actions.view(BATCH_SIZE, -1)
            next_observations = next_states.view(BATCH_SIZE, -1)

            # take rewards and dones for this agent only
            r = rewards[:, agent_id].unsqueeze_(1)
            dones = dones[:, agent_id].unsqueeze_(1)

            #---------------------update critic-------------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actions_target(next_states)
            actions_next = actions_next.view(BATCH_SIZE, -1)

            Q_targets_next = agent.critic_target(next_observations, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = r + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = agent.critic_local(observations, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            #----------------------update actor-------------------------------#
            agent.actor_optimizer.zero_grad()
            actions_local = self.actions_local(states, agent_id)
            actions_local = actions_local.view(BATCH_SIZE, -1)
            q_value_predicted = agent.critic_local(observations, actions_local)
            loss = -q_value_predicted.mean()
            loss.backward()
            agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        for agent in self.agents:
                agent.soft_update(agent.critic_local, agent.critic_target, TAU)
                agent.soft_update(agent.actor_local, agent.actor_target, TAU)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def actions_target(self, states):
        with torch.no_grad():
            actions = torch.empty(
                (BATCH_SIZE, len(self.agents), self.action_size),
                device=device)
            for idx, agent in enumerate(self.agents):
                actions[:,idx] = agent.actor_target(states[:,idx])
        return actions

    def actions_local(self, states, agent_id):
        actions = torch.empty(
            (BATCH_SIZE, len(self.agents), self.action_size),
            device=device)
        for idx, agent in enumerate(self.agents):
            action = agent.actor_local(states[:,idx])
            if not idx == agent_id:
                action.detach()
            actions[:,idx] = action
        return actions

    def actor_state_dict(self):
        return [agent.actor_local.state_dict() for agent in self.agents]

    def load_actor_state_dict(self, state_dict_list):
        for agent, state_dict in zip(self.agents, state_dict_list):
            agent.actor_local.load_state_dict(state_dict)

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            replay_buffer (obj): In MADDPG replay buffer is shared among all agents
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, num_agents, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, num_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        tempState = [e.state for e in experiences if e is not None]
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
