"""
Switch two types of exploration. Unsuccessuful trial. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class Actor(nn.Module):
    def __init__(self, 
                 value, 
                 n_states=5, 
                 n_actions=1, 
                 bidMin=.2, 
                 bidMax=5, 
                 hidden1=200, 
                 hidden2=100, 
                 constrain=True):
        """
        Actor is a neural network that takes states and input and action as output.
        The inner activation function is relu. The outer activation function is tanh. After which a linear transformation is performed to tranform this to [bid_min, self.value]
        The weights is initiated using xavier. The bias is initiated as 0. 
        """
        assert bidMax > bidMin
        assert bidMin > 0
        assert hidden1 > 0 and isinstance(hidden1, int)
        assert hidden2 > 0 and isinstance(hidden2, int)
        assert n_actions > 0 and isinstance(n_actions, int)
        assert n_states > 0 and isinstance(n_states, int)

        super().__init__()
        
        self.bidMin = bidMin
        self.bidMax = bidMax
        self.n_states = n_states
        self.n_actions = n_actions
        self.value = value
                
        if constrain:
            self.bidMax=self.value
            
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_states, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions),
            nn.Tanh()
        )
        
  
        for layer in self.children():
            self.__initWeights(layer)

    def __initWeights(self, m):
        if isinstance(m, torch.nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, states):
        # Input Shape: tensor(n_batches, n_states)
        # Output Shape: tensor(n_batches, n_actions)
        assert states.type().split(".")[-1] == 'FloatTensor'
        assert states.shape[-1] == self.n_states
        
        out = (self.linear_relu_stack(states) + 1) /  2 * (self.bidMax-self.bidMin) + self.bidMin
        return out


class Critic(nn.Module):
    def __init__(self, n_states=5, n_actions=1, hidden1=50, hidden2=20):
        """
        Critic is a neural network that takes (states, action) as input and Q-value as output.
        The inner activation function is relu.
        The weights is initiated using xavier. The bias is initiated as 0. 
        """
        assert hidden1 > 0 and isinstance(hidden1, int)
        assert hidden2 > 0 and isinstance(hidden2, int)
        assert n_actions > 0 and isinstance(n_actions, int)
        assert n_states > 0 and isinstance(n_states, int)

        super().__init__()
        
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.fc = nn.Linear(n_states, hidden1)
        self.relu = nn.ReLU()
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(hidden1+n_actions, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1)   
        )
        for layer in self.children():
            self.__initWeights(layer)

    def __initWeights(self, m):
        if isinstance(m, torch.nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, states, actions):
        # Input Shape: [tensor(n_batches, n_states), tensor(n_batches, n_actions)]
        # Output Shape: tensor(n_batches, 1)

        assert states.type().split(".")[-1] == 'FloatTensor'
        assert actions.type().split(".")[-1] == 'FloatTensor'
        assert states.shape[-1] == self.n_states
        assert actions.shape[-1] == self.n_actions
        
        out = self.fc(states)
        out = self.relu(out)
        out = self.linear_relu_stack(torch.cat([out, actions], dim=-1))
        return out


class Memory:
    """
    Memory is container of past experiences.
    """
    def __init__(self, n_states=5, n_actions=1):
        assert n_actions > 0 and isinstance(n_actions, int)
        assert n_states > 0 and isinstance(n_states, int)

        self.epoch = 0
        self.n_states = n_states
        self.n_actions = n_actions
        self.container = {}

    def append(self, states_old, actions, states_new, reward, epoch):
        """
        Push an individual experience into queue.
        """
        assert states_old.type().split(".")[-1] == 'FloatTensor'
        assert actions.type().split(".")[-1] == 'FloatTensor'
        assert states_new.type().split(".")[-1] == 'FloatTensor'
        assert states_old.shape[-1] == self.n_states
        assert states_new.shape[-1] == self.n_states
        assert reward.dim() == 1
        assert reward.type().split(".")[-1] == 'FloatTensor'
        assert epoch >= 0 and isinstance(epoch, int)

        self.epoch = epoch
        if epoch not in self.container:
            self.container[epoch] = []
        self.container[epoch].append((states_old, actions, states_new, reward))

    def sample_latestN(self, n, epoch):
        """
        fetch the last n experiences.
        """
        assert n > 0 and isinstance(n, int)
        assert epoch >= 0 and isinstance(epoch, int)
        assert epoch in self.container

        if len(self.container[epoch]) < n:
            print("Memory not enough. Return all!")

        states_old_ls, actions_ls, states_new_ls, reward_ls = zip(
            *self.container[epoch][-n:])
        states_old_ls = torch.stack(states_old_ls)
        actions_ls = torch.stack(actions_ls)
        states_new_ls = torch.stack(states_new_ls)
        reward_ls = torch.stack(reward_ls)

        return states_old_ls, actions_ls, states_new_ls, reward_ls
    
    def __len__(self):
        return len(self.container)
    
    def get_capacity(self, epoch):
        return len(self.container[epoch])
    
        
def syncronize(target, source, tau):
    """
    Syncronize target and source network, with parameter tau of EMA.
    """
    assert isinstance(target, nn.Module)
    assert isinstance(source, nn.Module)
    assert type(target) == type(source)

    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau)

class DDPG:
    def __init__(self, 
                 index, 
                 value, 
                 hidden1_actor,
                 hidden2_actor,
                 hidden1_critic,
                 hidden2_critic,
                 device,
                 n_states=5, 
                 n_actions=1, 
                 lr_actor=1E-4, 
                 lr_critic=1E-3, 
                 batch_size=5, 
                 gamma=0.95, 
                 tau=0.001,
                 bidMin=.2, 
                 bidMax=5, 
                 epoch=0,
                 constrain=True):
        assert tau > 0 and tau < 1 and isinstance(tau, float)
        assert n_states > 0 and isinstance(n_states, int)
        assert n_actions > 0 and isinstance(n_actions, int)
        assert lr_actor > 0 and lr_actor < 1 and isinstance(lr_actor, float)
        assert lr_critic > 0 and lr_critic < 1 and isinstance(lr_critic, float)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert gamma >= 0 and gamma < 1 and isinstance(gamma, float)
        assert epoch >= 0 and isinstance(epoch, int)
        assert index >= 0 and isinstance(index, int)
        assert value > 0 and isinstance(value, float)
        # assert hidden1 > 0 and isinstance(hidden1, int)
        # assert hidden2 > 0 and isinstance(hidden2, int)
        
        self.device = device
        self.tau = tau
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.gamma = gamma
        self.bidMin = bidMin
        self.bidMax = bidMax
        self.epoch = epoch
        self.index = index
        self.steps = 0
        self.value = value
        self.cache = []
        self.hidden1_actor = hidden1_actor
        self.hidden2_actor = hidden2_actor
        self.hidden1_critic = hidden1_critic
        self.hidden2_critic = hidden2_critic
        self.constrain = constrain
        self.memory = Memory(n_states=self.n_states, n_actions=self.n_actions)
        self.reset(epoch=0)
        
        syncronize(target=self.actor_target, source=self.actor_source, tau=1)
        syncronize(target=self.critic_target, source=self.critic_source, tau=1)
        
        if constrain:
            self.bidMax = self.value

    def act(self, states, base, is_train=True, requires_grad=True, override=None, epsilon_0=.5, beta=1E-4 *3, dynamic_explore=True):
        """
        Make a bid.
        """
        assert states.type().split(".")[-1] == "FloatTensor"
        assert states.shape[-1] == self.n_states
        assert override is None or isinstance(override, float)
        assert isinstance(epsilon_0, float) and epsilon_0 >=0 and epsilon_0 <=1
        assert isinstance(beta, float) and beta >=0 
         
        if override is not None:
            return torch.tensor([override]).to(self.device)
        
        # if not requires_grad:
        #     with torch.no_grad():
        #         actions = self.actor_source(states)
        # else:
        #     actions = self.actor_source(states)
        
        # assert actions.item() >= self.bidMin and actions.item() <= self.bidMax
        
        # if self.steps < 50:
        #     actions -= is_train * (torch.rand(size=(1, )) - .5) * 5
        # else:
        #     actions -= is_train * (torch.rand(size=(1, )) - .5) * 5 * torch.exp(torch.tensor(-(self.steps-50) / 5) ) * 1
        
        explore_uniform = False
        explore_normal = False

        if not dynamic_explore:
            self.lastReward = 1
            
        if torch.rand(size=(1,)).item() < epsilon_0 * (base ** torch.tensor(-beta * self.steps)).item():
            explore_normal = True
        
        if self.lastReward < 1E-2:
            explore_uniform = True

        if explore_uniform:
            actions = torch.rand(size=(1,)) * (self.bidMax-self.bidMin) + self.bidMin

        else:
            if not requires_grad:
                with torch.no_grad():
                    actions = self.actor_source(states)
                    if explore_normal:
                        actions += torch.normal(mean=torch.tensor(0.), std=torch.tensor(.5))
            else:
                actions = self.actor_source(states)
                if explore_normal:
                    actions += torch.normal(mean=torch.tensor(0.), std=torch.tensor(.5))
            
        actions = actions.clamp(self.bidMin, self.bidMax)
        
        self.steps += 1
        return actions.to(self.device)
    
    def update(self, loss_fun=nn.MSELoss()):
        """
        Batch-train value netowrk and policy network.
        Please check the algorithm of DDPG paper.
        """
        states_old_ls, actions_ls, states_new_ls, reward_ls = self.memory.sample_latestN(self.batch_size, epoch=self.epoch)
        self.lastReward = reward_ls[-1]
        
        with torch.no_grad():
            y_target = reward_ls + self.gamma * self.critic_target(states_new_ls, self.actor_target(states_new_ls))
        
        self.critic_source.zero_grad()
        loss_critic = loss_fun(self.critic_source(states_old_ls, actions_ls), y_target)
        loss_critic.backward()
        self.optimizer_critic.step()
        
        for i in range(3):
            self.actor_source.zero_grad()
            loss_actor = - self.critic_source(states_old_ls, self.actor_source(states_old_ls)).mean()
            loss_actor.backward()
            self.optimizer_actor.step()
        
        self.cache[-1].append((loss_actor.item(), loss_critic.item()))
        
        syncronize(target=self.actor_target, source=self.actor_source, tau=self.tau)
        syncronize(target=self.critic_target, source=self.critic_source, tau=self.tau)
    
    def reset(self, epoch):
        self.actor_source = Actor(n_states=self.n_states, n_actions=self.n_actions, bidMin=self.bidMin, bidMax=self.bidMax, value=self.value, hidden1=self.hidden1_actor, hidden2=self.hidden2_actor, constrain=self.constrain)
        self.actor_target = Actor(n_states=self.n_states, n_actions=self.n_actions, bidMin=self.bidMin, bidMax=self.bidMax, value=self.value, hidden1=self.hidden1_actor, hidden2=self.hidden2_actor, constrain=self.constrain)
        self.optimizer_actor = Adam(self.actor_source.parameters(), lr=self.lr_actor)

        self.critic_source = Critic(n_states=self.n_states, n_actions=self.n_actions, hidden1=self.hidden1_critic, hidden2=self.hidden2_critic)
        self.critic_target = Critic(n_states=self.n_states, n_actions=self.n_actions, hidden1=self.hidden1_critic, hidden2=self.hidden2_critic)
        self.optimizer_critic = Adam(self.critic_source.parameters(), lr=self.lr_critic)
        self.steps = 0
        
        self.lastReward = -1
        self.epoch=epoch
        self.actor_source.to(self.device)
        self.actor_target.to(self.device)
        self.critic_source.to(self.device)
        self.critic_target.to(self.device)


