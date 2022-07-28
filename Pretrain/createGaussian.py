import numpy as np
import os
from pathlib import Path
import seaborn as sns

"""
Excerpted from Michele's code.
"""
n_actions = 25
max_bid = 5
min_bid = .2
step = (max_bid - min_bid) / (n_actions - 1)
delta = .95

valuations = [5., 4., 3., 2., 1.]
bids = [min_bid + (i * step) for i in range(n_actions)]
valuation_actions = [np.argmin([abs(bid - val) for bid in bids]) for val in valuations]

def soft_max(np_array):
    """
    Compute soft-max values for each sets of scores in x.
    :param np_array:
    """

    e_x = np.exp(np.subtract(np_array, np.max(np_array)))
    return np.divide(e_x, e_x.sum())

def draw_from_gaussian_distribution(x, exp_value=0, variance=1):
    """
    Give exp_value and the variance creates a random distribution
    from where to draw an x value
    :param x:
    :param exp_value:
    :param variance:
    :return: f(x)
    """

    return 1 / np.sqrt(2*np.pi*variance) * np.exp(-np.square(x-exp_value)/(2 * variance))


def gaussian_init(vector_length, exp_value, variance):
    """
    Creates a vector of float numbers centered on exp_value with the desired variance,
    the output is normalized between [0, 1] interval through a softmax
    """
    row = [draw_from_gaussian_distribution(act,
                                           exp_value=exp_value,
                                           variance=variance)
           for act in range(vector_length)]

    norm_row = soft_max(np.array(row))

    return norm_row

clickRates_ls = (20, 10, 5, 2, 0)
revenue_max = clickRates_ls[0] * (max_bid - min_bid)

norm_row_ls = [gaussian_init(vector_length=n_actions,
                                         exp_value=valuation_actions[i],
                                         variance=n_actions)  * revenue_max / (1-delta) for i in range(5)]

tabSavePath="/home/3068020/GameTheory/q_learning_paper/q_larker/qDiscrete/Normal"
Path(tabSavePath).mkdir(parents=True, exist_ok=True)

for i in range(5):
    np.save(os.path.join(tabSavePath, "compressedQmat" + str(i) + ".npy"), norm_row_ls[i])

sns.heatmap(norm_row_ls[::-1], yticklabels=range(1, 6)).set_title("Calvano Qmat")


