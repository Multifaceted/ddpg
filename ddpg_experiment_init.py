"""
DDPG experiment with initialization.
"""


import inspect
import numpy as np
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)  

from ddpg import DDPG, AuctionContEnv
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torch
import threading
import time
from DDPGvanilla_gpu import Actor

def mkDir(save_to):
    from pathlib import Path
    import os
    
    Path(os.path.join(save_to)).mkdir(parents=True, exist_ok=True)


def save_result(bidders_ls, dir_name="default"):
    import os
    import pickle
    from pathlib import Path

    Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    for bidder in bidders_ls:
        file_name = os.path.join(dir_name, str(bidder.index))
        
        with open(file_name, 'wb') as f:
            pickle.dump(bidder.memory, f)
            
def run_experiment(params, epoch, device, bidders_ls=None, reset=False, override_ls=None):

    writer = SummaryWriter(log_dir=os.path.join(params["save_to"], "TensorboadEpoch"+str(epoch)))
    if bidders_ls is None:
        bidders_ls = [DDPG(index=i, 
                        value=params["valuations_ls"][i], 
                        n_states=params["n_players"], 
                        n_actions=1, 
                        lr_actor=params["lr_actor"][i], 
                        lr_critic=params["lr_critic"][i], 
                        batch_size=params["batch_size"], 
                        gamma=params["delta"], 
                        tau=params["sync_rate"],
                        bidMin=params["min_bid"], 
                        bidMax=params["max_bid"], 
                        # bidMax=params["valuations_ls"][i], 
                        epoch=epoch,
                        hidden1_actor=params["hidden1_actor"],
                        hidden2_actor=params["hidden2_actor"],
                        hidden1_critic=params["hidden1_critic"],
                        hidden2_critic=params["hidden2_critic"],
                        constrain=params["constrain"],
                        device=device) for i in range(params["n_players"])]
        
    if reset:
        [bidder.reset(epoch=epoch) for bidder in bidders_ls]
        
    env = AuctionContEnv(n_bidders=params["n_players"], 
                         clickRates_ls=params["clickRates_ls"], 
                         valuations_ls=params["valuations_ls"], 
                         bidMin=.2, 
                         bidMax=5, 
                         seed=None,
                         device=device
                         )
    if override_ls is None:
        override_ls = [None] * params["n_players"]
        
    [bidder.cache.append([]) for bidder in bidders_ls]
    for round in range(params["n_rounds"]):
        states_old_ls = env.states  # randomly initialized states
        states_old_ls = states_old_ls.to(device)
        actions_ls = torch.cat([bidder.act(states_old_ls, 
                                           is_train=True, 
                                           requires_grad=False, 
                                           override=override_ls[idx],
                                           beta=params["beta"],
                                           epsilon_0=params["epsilon_0"],
                                           base=params["base"],
                                           dynamic_explore=params["dynamic_explore"]) for idx, bidder in enumerate(bidders_ls)]) # take actions (make bids)
        actions_ls = actions_ls.to(device)
        states_new_ls = actions_ls # update states
        rewards_ls = env.step(actions_ls)[1] # environment gives rewards
        [bidders_ls[i].memory.append(states_old_ls, actions_ls[[i]], states_new_ls, rewards_ls[[i]], epoch=epoch) for i in range(params["n_players"])] # commit experience into memory
        if round % 100 == 0:
            print("Round: ", round)
            print("Actions: ", actions_ls)
            print("Rewards: ", rewards_ls)

        if round % 1000 == 0:
            for i in range(params["n_players"]):
                np.savetxt(os.path.join(params["save_to"], "losses_epoch"+str(epoch)+"player"+str(i)), bidders_ls[i].cache[-1])

        [bidder.update(writer, round) for i, bidder in enumerate(bidders_ls) if override_ls[i] is None] # learn from memory
        
        for agent in range(params["n_players"]):
            writer.add_scalar("Action"+str(agent), actions_ls[agent].item(), round)
            writer.add_scalar("Reward"+str(agent), rewards_ls[agent].item(), round)

    print("Saving Result of epoch:", epoch)
    saveMatrix(bidders_ls, save_to=os.path.join(params["save_to"], str(epoch)))
    return bidders_ls

class ExperimentThread(threading.Thread):
    def __init__(self, threadID, maxThread, **kwargs):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.maxThread = maxThread
      self.kwargs = kwargs

    def run(self):
        global thread_counter
        
        while thread_counter >= self.maxThread:
            time.sleep(30)
            print("Thread", self.threadID, "is waiting...")
        thread_counter += 1
        run_experiment(**self.kwargs)
        thread_counter -= 1
        print("Thread", str(self.threadID), "is finished. thread_counter:", str(thread_counter))


def run_experimentMultiple(params, maxThread, override_ls=None):
    # bidders_ls = [DDPG(index=i, 
    #                 value=params["valuations_ls"][i], 
    #                 n_states=params["n_players"], 
    #                 n_actions=1, 
    #                 lr_actor=params["lr_actor"], 
    #                 lr_critic=params["lr_critic"], 
    #                 batch_size=params["batch_size"], 
    #                 gamma=params["delta"], 
    #                 tau=params["sync_rate"],
    #                 bidMin=params["min_bid"], 
    #                 bidMax=params["max_bid"], 
    #                 # bidMax=3,
    #                 epoch=params["epoch"],
    #                 hidden1_actor=params["hidden1_actor"],
    #                 hidden2_actor=params["hidden2_actor"],
    #                 hidden1_critic=params["hidden1_critic"],
    #                 hidden2_critic=params["hidden2_critic"],
    #                 constrain=params["constrain"],
    #                 device=device) for i in range(params["n_players"])]
    
    for epoch in range(params["epoch"]):
        # bidders_ls = run_experiment(params, bidders_ls, override_ls=override_ls, reset=True)
        ExperimentThread(threadID=epoch, maxThread=maxThread, params=params, epoch=epoch, device=devices_ls[epoch%(n_devices-1)+1] if n_devices>0 else "cpu", override_ls=override_ls).start()
        # ExperimentThread(threadID=epoch, maxThread=maxThread, params=params, epoch=epoch, device=devices_ls[1] if n_devices>0 else "cpu", override_ls=override_ls).start()

    return 
    
    # save_result(bidders_ls, dir_name=params["save_to"])
def fetch_results(bidders_ls):
    import numpy as np

    actions_mat = [[ [data[1].item() for data in epoch_data] for epoch_data in bidder.memory.container.values()] for bidder in bidders_ls]
    # actions_avg_ls = [np.array(action).mean(axis=0) for action in actions_ls]
    
    rewards_mat = [[ [data[3].item() for data in epoch_data] for epoch_data in bidder.memory.container.values()] for bidder in bidders_ls]
    # rewards_avg_ls = [np.array(reward).mean(axis=0) for reward in rewards_ls]

    return np.swapaxes(np.array(actions_mat), 0, 1), np.swapaxes(np.array(rewards_mat), 0, 1)

def saveMatrix(bidders_ls, save_to):
    import pandas as pd
    import os
    import json

    mkDir(save_to)
    actions_mat, rewards_mat = fetch_results(bidders_ls)

    actions_df = pd.DataFrame(actions_mat.T[..., 0])
    rewards_df = pd.DataFrame(rewards_mat.T[..., 0])


    print("actions_df address:", os.path.join(save_to, "actions.csv"))
    actions_df.to_csv(os.path.join(save_to, "actions.csv"))
    rewards_df.to_csv(os.path.join(save_to, "rewards.csv"))
    json.dump(params, open(os.path.join(save_to, "Params.json"), "w"), sort_keys=True, indent=4)

def saveDDPG(bidders_ls, indexes_ls=[0, 1, 2, 3, 4], experimentName="defaultExperiment"):
    import pickle
    import os
    import json
    
    mkDir(experimentName)
    for index, bidder in enumerate(bidders_ls):
        if index in indexes_ls:
            pickle.dump(bidder, open(os.path.join("expResult", experimentName, "Agent"+str(index)), "wb"))
        json.dump(params, open(os.path.join("expResult", experimentName, "Params.json"), "w"), sort_keys=True, indent=4)
    
def fetch_cache(caches_ls):
    import numpy as np
    
    caches_ls = [np.array(cache).mean(axis=0).T for cache in caches_ls]
    losses_actor_ls = [cache[0] for cache in caches_ls]
    losses_critic_ls = [cache[1] for cache in caches_ls]
    return losses_actor_ls, losses_critic_ls
    
if __name__ == '__main__':
    
    path_model = "Model/CalvanoLessTrain2"
    exec(open("DDPGvanilla_gpu.py").read())

    params2 = {
        "n_players": 5,
        "min_bid": .2,
        "max_bid": 5,
        "delta": .95,
        "n_rounds": 200000,
        "lr_actor": [1.E-2] * 3 + [1.E-2] * 2,
        "lr_critic": [1.E-2] * 3 + [1.E-1] * 2,
        "sync_rate": .9, 
        "batch_size": 20,
        "epoch": 5,
        "valuations_ls": (5., 4., 3., 2., 1.),
        "clickRates_ls": (20, 10, 5, 2, 0),
        "save_to": "expResult/CalvanoLooseActor",
        "hidden1_actor": 50,
        "hidden2_actor": 30, # Please change this
        "hidden1_critic": 120,
        "hidden2_critic": 80, # Please change this
        "constrain": False,
        "beta": 1E-2,
        "epsilon_0": .99,
        "base": 1.01,
        "maxThread": 9,
        "dynamic_explore": True
    }
    n_devices = torch.cuda.device_count()
    if n_devices == 0:
        devices_ls = ["cpu"]
    else:
        # devices_ls = [torch.device('cuda:'+str(i)) for i in range(n_devices)]
        devices_ls = [torch.device('cuda:'+str(i)) for i in range(0, 3)]

    thread_counter = 0
    # device = torch.device('cpu')

    device = devices_ls[0]

    for epoch in range(params2["epoch"]):
        params = params2.copy()
        params["epoch"] = epoch
        actors_ls = [torch.load(os.path.join(path_model, "actor" + str(i) + ".pt")) for i in range(5)]
        # actors_ls = [Actor(value=v, constrain=False) for v in range(5, 0, -1)]
        critics_ls = [torch.load(os.path.join(path_model, "critic" + str(i) + ".pt")) for i in range(5)]

        bidders_ls = [DDPG(index=i, 
                                value=params["valuations_ls"][i], 
                                n_states=params["n_players"], 
                                n_actions=1, 
                                lr_actor=params["lr_actor"][i], 
                                lr_critic=params["lr_critic"][i], 
                                batch_size=params["batch_size"], 
                                gamma=params["delta"], 
                                tau=params["sync_rate"],
                                bidMin=params["min_bid"], 
                                bidMax=params["max_bid"], 
                                # bidMax=params["valuations_ls"][i], 
                                epoch=epoch,
                                hidden1_actor=params["hidden1_actor"],
                                hidden2_actor=params["hidden2_actor"],
                                hidden1_critic=params["hidden1_critic"],
                                hidden2_critic=params["hidden2_critic"],
                                constrain=params["constrain"],
                                device=device) for i in range(params["n_players"])]
        for i in range(params["n_players"]):
            bidders_ls[i].actor_source = actors_ls[i].to(device)
            bidders_ls[i].actor_target = actors_ls[i].to(device)
            bidders_ls[i].critic_source = critics_ls[i].to(device)
            bidders_ls[i].critic_target = critics_ls[i].to(device)

        run_experiment(params, epoch=epoch, device=device, bidders_ls=bidders_ls, reset=False, override_ls=None)

    # actions_avg_ls, rewards_avg_ls = fetch_results(bidders_ls)
    # caches_ls = [bidder.cache for bidder in bidders_ls]
    # losses_actor_ls, losses_critic_ls = fetch_cache(caches_ls)
    # plot_actions(actions_avg_ls, valuations_ls = [5,4,3,2,1], n_epochs=n_epochs, save=True, experimentName=params["ExperimentName"])
    # plot_rewards(rewards_avg_ls, valuations_ls = [5,4,3,2,1], n_epochs=n_epochs, save=True, experimentName=params["ExperimentName"])
    # plot_losses(losses_actor_ls, valuations_ls = [5,4,3,2,1], n_epochs=n_epochs, plotName="actor", save=True, indexes_ls=[2], experimentName=params["ExperimentName"])
    # plot_losses(losses_critic_ls, valuations_ls = [5,4,3,2,1], n_epochs=n_epochs, plotName="critic", save=True, indexes_ls=[2], experimentName=params["ExperimentName"])
    # plotExploration(epsilon_0=.5, beta=1E-3, n_rounds=10000*2, save=True, experimentName=params["ExperimentName"])