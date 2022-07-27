import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
from pathlib import Path
import json

def saveQmat(index):
    path = '../qDiscrete/' 
    mat = torch.load(path + "averageQmat" + str(index) + ".pt")
    Qvalue = mat[0, 0, 0, 0, 0, ...].numpy()
    np.save(path + "compressedQmat" + str(index), Qvalue)

def trainAgent(index, grid_ls, params, loss, optimizer, path, **kwargs):
    import torch
    import numpy as np

    path = path
    Qvalue = np.load(path + "compressedQmat" + str(index) + ".npy")

    Pmat = torch.cartesian_prod(*([torch.tensor(grid_ls)] * (params["n_players"]) )).numpy()
    y = np.repeat(grid_ls[Qvalue.argmax()], Pmat.shape[0])
    
    training_set = Dataset(Pmat.astype(np.float32), y.astype(np.float32))
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True, pin_memory=True)

    trainActor(training_generator=training_generator, params=params, index=index, loss=loss, optimizer=optimizer,
                **kwargs)

def trainActor(actor, 
                training_generator, 
                loss, 
                optimizer, 
                params, 
                n_epochs, 
                device,
                dir_save,
                index,
                **kwargs):

    import os
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    
    writer = SummaryWriter(log_dir=os.path.join(dir_save, "ActorTensorboard"+str(index)))
    loss_ls = np.zeros(100)
    n_steps = 0
    for epoch in (pbar:= tqdm(range(n_epochs))):
        for X, y in training_generator:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            output = actor(X)
            Loss = loss(output.view(-1), y.view(-1))  
            Loss.backward()
            optimizer.step()
            n_steps += 1
            loss_ls[n_steps%100] = Loss.item()
            if n_steps%100 == 0:
                pbar.set_description("Loss %s" % loss_ls.mean())
                writer.add_scalar("Loss/train", Loss.item(), n_steps)
        with open(os.path.join(dir_save, "actor" + str(index) +".pt"), "wb") as fp:
            torch.save(actor, fp)
    writer.close()

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, y):
        'Initialization'
        self.data_X = X
        self.data_y = y

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_y)

  def __getitem__(self, index):
        'Generates one sample of data'

        return self.data_X[index], self.data_y[index]



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


if __name__ == '__main__':

    from torch import nn
    import torch
    import numpy as np
    from torch import optim
    params = {
    "n_players": 5,
    "min_bid": .2,
    "max_bid": 5,
    "delta": 0.9,
    "n_rounds": 5000,
    "lr_actor": 1E-3,
    "lr_critic": 1E-2,
    "sync_rate": .9, 
    "batch_size": 1,
    "epoch": 0,
    "valuations_ls": (5., 4., 3., 2., 1.),
    "clickRates_ls": (20, 10, 5, 2, 0),
    "save_to": "../result/test",
    "hidden1_actor": 50,
    "hidden2_actor": 30, # Please change this
    "hidden1_critic": 120,
    "hidden2_critic": 80, # Please change this
    "constrain": False,
    "ExperimentName": "post_exp_unconstrained"
}

    params_train={
                  "n_epochs": 1,
                  "dir_save": "Model/CalvanoLessTrain2",
                  "grid_ls": list(np.linspace(.2, 5, 25, endpoint=True)),
                  "index": 4,
                  "lr": 1E-3

    }
    loss = nn.MSELoss()
    device = torch.device("cuda:0")
    actor = Actor(n_states=params["n_players"], n_actions=1, hidden1=params["hidden1_actor"], hidden2=params["hidden2_actor"], constrain=False, value=-1).to(device)
    optimizer = optim.Adam(actor.parameters(), params_train["lr"])
    Path(params_train["dir_save"]).mkdir(parents=True, exist_ok=True)
    trainAgent(loss=loss, optimizer=optimizer, params=params, actor=actor, device=device, path='../qDiscrete/', **params_train)
    with open(os.path.join(params_train["dir_save"], "train_params_actor" + str(params_train["index"]) + ".json"), 'w') as fp:
        json.dump(params_train, fp)
