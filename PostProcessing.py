"""
This file is created during the visualization and statistical analysis. 
This script was intended to solve various immediate request and thus are not well structured.
It is not discarded in case someone may find it useful.
"""
import inspect
import numpy as np
import os

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
device = "cpu"

params = {
    "n_players": 5,
    "min_bid": .2,
    "max_bid": 5,
    "delta": .3,
    "n_rounds": 50000,
    "lr_actor": 1E-3,
    "lr_critic": 1E-2,
    "sync_rate": .9,
    "batch_size": 20,
    "epoch": 9,
    "valuations_ls": (5., 4., 3., 2., 1.),
    "clickRates_ls": (20, 10, 5, 2, 0),
    "save_to": "../result/test",
    "hidden1_actor": 150,
    "hidden2_actor": 80,  # Please change this
    "hidden1_critic": 500,
    "hidden2_critic": 200,  # Please change this
    "constrain": False,
    "beta": 1E-3,
    "epsilon_0": .99,
    "base": 1.1,
    "ExperimentName": "dynamic_reverse2",
    "maxThread": 9,
    "dynamic_explore": True
}


def read_result(dir_name, n_bidders, epoch):
    """
    Read result from agent pickle file and return bidder list.
    """
    import os
    import pickle

    bidders_ls = []
    for e in range(epoch):
        print("fetching epoch:", str(e))
        for i in range(n_bidders):
            print("fetching bidder:", str(i))
            with open(os.path.join(dir_name, str(e), "Agent"+str(i)), 'rb') as f:
                agent = pickle.load(f)
                if e == 0:
                    bidders_ls.append(agent)
                else:
                    bidders_ls[i].memory.container[e] = agent.memory.container[e]
    return bidders_ls


def fetch_results(bidders_ls):
    """
    Fetch action list and reward list from bidder list.
    """
    import numpy as np

    actions_mat = [[[data[1].item() for data in epoch_data]
                    for epoch_data in bidder.memory.container.values()] for bidder in bidders_ls]
    # actions_avg_ls = [np.array(action).mean(axis=0) for action in actions_ls]

    rewards_mat = [[[data[3].item() for data in epoch_data]
                    for epoch_data in bidder.memory.container.values()] for bidder in bidders_ls]
    # rewards_avg_ls = [np.array(reward).mean(axis=0) for reward in rewards_ls]

    return np.swapaxes(np.array(actions_mat), 0, 1), np.swapaxes(np.array(rewards_mat), 0, 1)


def plotPipeline(dir_name, ma=500, ci=.95):
    """
    Plot from actions.csv file
    """
    import numpy as np
    import scipy.stats
    import matplotlib.pyplot as plt

    df = read_resultCSV(dir_name, "actions.csv", ma)
    n_bidders = df.columns.nunique()
    n_epochs = len(df.columns) // n_bidders

    n_sigmas = scipy.stats.t(df=n_epochs-1).ppf(1-(1-ci)/2)

    f, ax = plt.subplots(1, 2, figsize=(20, 10))

    df_group = df.groupby(df.columns, axis=1)
    df_lb = df_group.mean() - n_sigmas * df_group.std() / np.sqrt(n_epochs)
    df_ub = df_group.mean() + n_sigmas * df_group.std() / np.sqrt(n_epochs)
    df_avg = df_group.mean()

    plt.suptitle("Average epoch " + str(n_epochs) + " CI " + str(ci))
    ax[0].set_title("actions")

    for bidder in df_avg.columns:
        ax[0].fill_between(df_avg.index, df_lb[bidder],
                           df_ub[bidder], alpha=.3)
        ax[0].plot(df_avg.index, df_avg[bidder],
                   label="Value " + str(n_bidders - int(bidder)))
        ax[0].legend()

    df = read_resultCSV(dir_name, "rewards.csv", ma)
    n_bidders = df.columns.nunique()
    n_epochs = len(df.columns) // n_bidders

    n_sigmas = scipy.stats.t(df=n_epochs-1).ppf(1-(1-ci)/2)

    df_group = df.groupby(df.columns, axis=1)
    df_lb = df_group.mean() - n_sigmas * df_group.std() / np.sqrt(n_epochs)
    df_ub = df_group.mean() + n_sigmas * df_group.std() / np.sqrt(n_epochs)
    df_avg = df_group.mean()

    ax[1].set_title("rewards")

    for bidder in df_avg.columns:
        ax[1].fill_between(df_avg.index, df_lb[bidder],
                           df_ub[bidder], alpha=.3)
        ax[1].plot(df_avg.index, df_avg[bidder],
                   label="Value " + str(n_bidders - int(bidder)))
        ax[1].legend()

    f.savefig(dir_name + "/actions_rewards_plot.png", dpi=300)


def statisticsPipeline(dir_name, ma=500, ci=.95):
    """
    Fetch revenue and its standard edeviation, rate of efficiency and equilibrium and print them out.
    """
    import numpy as np
    import scipy.stats
    import os
    import sys

    df = read_resultCSV(dir_name, "actions.csv", ma)
    n_bidders = df.columns.nunique()
    n_epochs = len(df.columns) // n_bidders

    actions_ls = np.split(df.tail(1), n_epochs, axis=1)
    actions_ls = list(map(np.squeeze, actions_ls))

    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    rate_efficiency = np.mean(list(map(checkEfficiency, actions_ls)))
    rate_equilibrium = np.mean(list(map(checkEquilibrium, actions_ls)))
    sys.stdout = sys.__stdout__

    revenues_ls = list(map(computeRevenue, actions_ls))

    print("actions:", np.mean(actions_ls, axis=0))
    print("rate of efficiency:", rate_efficiency)
    print("rate of equilibrium:", rate_equilibrium)
    n_sigmas = scipy.stats.t(df=n_epochs-1).ppf(1-(1-ci)/2)
    revenue = np.mean(revenues_ls)
    revenue_std = np.std(revenues_ls) / np.sqrt(n_epochs)
    print("revenue:")
    print(revenue, (revenue - n_sigmas*revenue_std, revenue + n_sigmas*revenue_std))


def SuperPipeline(dir_name, ma=500, ci=.95):
    """
    Combind plotPipeline and statisticPipeline
    """
    import os
    for folder in os.listdir(dir_name):
        print(folder)
        plotPipeline(os.path.join(dir_name, folder))
        statisticsPipeline(os.path.join(dir_name, folder))


def read_resultCSV(dir_name, name, ma=500):
    """
    Auxiliary function for statisticsPipeline
    """
    from pathlib import Path
    import os
    import pandas as pd

    p = Path(dir_name)
    folders = os.walk(dir_name)
    dfs_ls = [pd.read_csv(os.path.join(epoch, name), index_col=0).reset_index(
        drop=True) for epoch in p.iterdir() if epoch.is_dir() and not "Tensorboad" in epoch.name]
    df = pd.concat(dfs_ls, axis=1).rolling(ma).mean()

    return df


def compute_revenueStatistics(actions_df, n_sigmas):
    """
    Computer revenue statistics.
    """
    import pandas as pd
    import numpy as np

    n_bidders = pd.to_numeric(actions_df.columns).max() + 1
    n_epochs = actions_df.shape[1] // n_bidders

    actions_last = np.split(actions_df.tail(1), n_epochs, axis=1)
    avg = np.mean([computeRevenue(actions_last[i].squeeze())
                  for i in range(n_epochs)])
    dev = np.std([computeRevenue(actions_last[i].squeeze())
                 for i in range(n_epochs)])
    dev_sampleMean = dev / np.sqrt(n_epochs)
    return avg, dev_sampleMean, avg - n_sigmas * dev_sampleMean, avg + n_sigmas * dev_sampleMean


def fetch_last(actions_mat, rewards_mat, ma=500):
    """
    fetch last actions by rolling window.
    """
    import pandas as pd
    import numpy as np

    n_epochs = actions_mat.shape[0]
    n_bidders = actions_mat.shape[1]
    res_actions_mat = np.zeros(shape=(n_epochs, n_bidders))
    res_rewards_mat = np.zeros(shape=(n_epochs, n_bidders))
    for epoch in range(n_epochs):
        res_actions_mat[epoch] = pd.DataFrame(
            actions_mat[epoch]).T.rolling(ma).mean().iloc[-1, :]
        res_rewards_mat[epoch] = pd.DataFrame(
            rewards_mat[epoch]).T.rolling(ma).mean().iloc[-1, :]

    return res_actions_mat, res_rewards_mat


def fetch_cache(caches_ls):
    import numpy as np
    """
    fetch actor losses and critic losses from caches.
    """

    caches_ls = [np.array(cache).mean(axis=0).T for cache in caches_ls]
    losses_actor_ls = [cache[0] for cache in caches_ls]
    losses_critic_ls = [cache[1] for cache in caches_ls]
    return losses_actor_ls, losses_critic_ls


def fetch_averageActions(actions_df):
    """
    Fetch average actgions.
    """
    return actions_df.groupby(actions_df, axis=1).mean()


def plot_actions(actions_ls, valuations_ls, n_epochs=0, save=False, experimentName="defaultExperiment", plotName="default"):
    """
    Plot actions.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if isinstance(actions_ls, pd.DataFrame):
        actions_ls = actions_ls.groupby(
            actions_ls.columns, axis=1).mean().T.to_numpy()
    plt.clf()
    for index, actions, value in zip(range(len(actions_ls)), actions_ls, valuations_ls):
        plt.plot(actions, label="Player_value_" + str(value))
    n_rounds = len(actions_ls[0])
    plt.xlabel('n_rounds')
    plt.ylabel('bids')
    plt.title("Avg Plot of Epoch " + str(n_epochs))
    plt.legend(loc="upper left")
    plt.hlines(y=1, xmin=0, xmax=n_rounds,
               linestyles="dashdot", color="#acacac")
    plt.hlines(y=2, xmin=0, xmax=n_rounds,
               linestyles="dashdot", color="#acacac")
    plt.hlines(y=3, xmin=0, xmax=n_rounds,
               linestyles="dashdot", color="#acacac")
    plt.hlines(y=4, xmin=0, xmax=n_rounds,
               linestyles="dashdot", color="#acacac")
    plt.hlines(y=5, xmin=0, xmax=n_rounds,
               linestyles="dashdot", color="#acacac")
    if save:
        mkDir(experimentName)
        plt.savefig(os.path.join("expResult", experimentName,
                    "actionsPlot_" + plotName))
    else:
        pass


def mkDir(experimentName):
    """
    Axuliary function for making directory.
    """
    from pathlib import Path
    import os

    Path(os.path.join("expResult", experimentName)).mkdir(
        parents=True, exist_ok=True)


def plot_rewards(rewards_ls, valuations_ls, n_epochs=0, save=False, experimentName="defaultExperiment", plotName="defaultRewards"):
    """
    Plot rewards.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if isinstance(rewards_ls, pd.DataFrame):
        rewards_ls = rewards_ls.groupby(
            rewards_ls.columns, axis=1).mean().T.to_numpy()
    plt.clf()
    for index, actions, value in zip(range(len(rewards_ls)), rewards_ls, valuations_ls):
        plt.plot(actions, label="Player_value_" + str(value))
    n_rounds = len(rewards_ls[0])
    plt.xlabel('n_rounds')
    plt.ylabel('bids')
    plt.title("Avg Plot of Epoch " + str(n_epochs))
    plt.legend(loc="upper left")

    if save:
        mkDir(experimentName)
        plt.savefig(os.path.join("expResult", experimentName,
                    "actionsPlot_" + plotName))
    else:
        pass


def plot_losses(losses_ls, valuations_ls, plotName, n_epochs=0, save=False, experimentName="defaultExperiment", indexes_ls=[0, 1, 2, 3, 4]):
    """
    Plot losses.
    """
    import matplotlib.pyplot as plt
    import os

    plt.clf()
    for index, losses in enumerate(losses_ls):
        if index in indexes_ls:
            plt.plot(losses, label="Player_value_" + str(valuations_ls[index]))
    plt.xlabel('n_rounds')
    plt.ylabel('losses')
    plt.title("Avg Plot of Epoch " + str(n_epochs) + "_" + plotName)
    plt.legend(loc="upper left")
    if save:
        mkDir(experimentName)
        plt.savefig(os.path.join(
            "expResult", experimentName, "lossesPlot_" + plotName))
    else:
        plt.show()


def saveDDPG(bidders_ls, params, indexes_ls=[0, 1, 2, 3, 4], experimentName="defaultExperiment"):
    """
    Save bidders to pickle file and dump parameters.
    """
    import pickle
    import os
    import json

    mkDir(experimentName)
    for index, bidder in enumerate(bidders_ls):
        if index in indexes_ls:
            pickle.dump(bidder, open(os.path.join(
                "expResult", experimentName, "Agent"+str(index)), "wb"))
        json.dump(params, open(os.path.join("expResult", experimentName,
                  "Params.json"), "w"), sort_keys=True, indent=4)


def plotExploration(epsilon_0, beta, n_rounds, save=False, experimentName="defaultExperiment"):
    """
    Plot exponential decay.
    """
    import matplotlib.pyplot as plt
    import torch

    explo_rate = [epsilon_0 * torch.exp(torch.tensor(-beta * step))
                  for step in range(n_rounds)]
    plt.clf()
    plt.plot(explo_rate)
    plt.xlabel('n_rounds')
    plt.ylabel('epsilon')
    plt.title('epsilon of epsilon-greedy')

    if save:
        mkDir(experimentName)
        plt.savefig(os.path.join("expResult", experimentName, "annihilation"))
    else:
        plt.show()


def checkEquilibrium(actions_ls):
    """
    Check equilibrium.
    """
    currentdir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    os.sys.path.insert(0, parentdir)

    from copy import deepcopy
    from ddpg import AuctionContEnv
    import torch

    actions_ls = torch.tensor(deepcopy(actions_ls), dtype=torch.float32)
    epsilon = 1E-5

    if 0 in actions_ls.sort()[0].diff():
        print("same bids!")
        return False

    env_test = AuctionContEnv(device="cpu",
                              n_bidders=params["n_players"],
                              clickRates_ls=params["clickRates_ls"],
                              valuations_ls=params["valuations_ls"],
                              bidMin=1E-5,
                              bidMax=100)

    _,  rewards_ls, _ = env_test.step(actions_ls)

    if (rewards_ls < 0).sum().item() > 0:
        print("negative rewards", rewards_ls)
        return False

    for self_idx in range(env_test.n_bidders):
        for other_idx in range(env_test.n_bidders):

            if self_idx == other_idx:
                continue

            actions_new_ls = deepcopy(actions_ls)
            actions_new_ls[self_idx] = actions_ls[other_idx] + epsilon
            _, rewards_new_ls, _ = env_test.step(actions_new_ls)
            if rewards_new_ls[self_idx] > rewards_ls[self_idx]:
                print("old rewards:", rewards_ls)
                print("better rewards", rewards_new_ls)
                print("better actions", actions_new_ls)
                return False
    return True


def checkEfficiency(actions_ls):
    import numpy as np
    """
    Check Efficiency. True if efficient
    """
    return np.array_equal(np.sort(actions_ls)[::-1], actions_ls)


def computeRevenue(actions_ls):
    """
    Auxiliary function for computeRevenueStatistics
    """
    price_ls = np.roll(np.sort(actions_ls), shift=1)
    price_ls[0] = 0
    return np.sum(price_ls * np.sort(params["clickRates_ls"]))



def plotIndividual(dir_name, name, ma, idx):
    """
    Plot individual runs.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    df = read_resultCSV(dir_name, name, ma)
    plt.subplots(2)
    plot_actions(
        df.iloc[:,  [x for x in np.arange(5) + 5*idx]], [5, 4, 3, 2, 1])
    # plot_rewards(df.iloc[:,  [x for x in np.arange(5) + 5*idx ]], [5, 4, 3, 2, 1])
    plt.title("idx: "+str(idx) + " ma: " + str(ma))
