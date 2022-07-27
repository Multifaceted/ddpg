## About The Project

In the discrete-action case of a multi-agent auction setting, Calvano Initialization was attempted with success. This project seeks to showcase its counterpart in the continuous-action setting. As the algorithm applied in the discrete case is Q-learning with Q-function
in the matrix form, the counterpart in the continuous case is naturally Deep Deterministic Policy Gradient. In continuous case, without pre-train, within limited steps of exploration (10K), we observe that the agents are able to figure out the order on average, but not the Nash Equilibrium, especially for individual runs. In this case, Calvano Initialization will help agents to converge faster and reach Nash Equilibrium.


<!-- GETTING STARTED -->
## Getting Started
  ```sh
  python ddpg_experiment_init.py
  ```

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

Please check environment.yml file in the repo. The code is deployed using pytorch and tensorboard. 


<!-- USAGE EXAMPLES -->
## Content
The code is based on classical ddpg algorithm with modifications and initializations. It is mainly composed of three parts: Pretrain, Experiment Environment and DDPG Models. 

* auctionContinuous_env.py  
Continuous action environment

* ddpg_experiment_init.py  
ddpg with initialization

* ddpg_experiment.py
ddpg without initialization

* DDPGvanilla.py  
DDPG on CPU

* DDPGvanilla_gpu.py  
DDPG on GPU

* createPretrainActor.py  
Create pre-trained actor neural network based on a discrete Q matrix.

* createPretrainCritic.py  
Create pre-trained creitic neural network based on a discrete Q matrix.

* createGaussian.py  
Create Gaussian Initialization matrix.

* PostProcessing.py  
Visualization and statistical functions for post analysis.

* environment.yml  
Environmental requirement

## Common Variables

    "n_players": number of players, usually 5
    "min_bid": minimal bid value, usually 0.2
    "max_bid": maximal bid value, usually 5 if not constrained
    "delta": discounted factor for accumulated reward
    "n_rounds": number of rounds
    "lr_actor": learning rate of actor neural network
    "lr_critic": learning rate of critical neural network
    "sync_rate": syncronizing rate of two copies of neural network (Doubel Q learning)
    "batch_size": number of training experiences fecthed from memory buffer
    "epoch": number of runs
    "valuations_ls": valuations of bidders, usually (5., 4., 3., 2., 1.)
    "clickRates_ls": click rates of ad slots, usually (20, 10, 5, 2, 0)
    "save_to": directory to which the result is saved
    "hidden1_actor": first hidden layer of actor neural netowrk
    "hidden2_actor": second hidden layer of actor neural netowrk
    "hidden1_critic": first hidden layer of critic neural netowrk
    "hidden2_critic": second hidden layer of critic neural netowrk
    "constrain": True if the actions is contrained
    "ExperimentName": Experiment Name (related to saving directory)
    "grid_ls": grid list, e.g. list(np.linspace(.2, 5, 25, endpoint=True)) for 25 actions
    "index": 0 for bidder value 5, 4 for bidder value 1.
}

## Pre-trained Models, Qmat and Result
https://bocconi-my.sharepoint.com/:f:/g/personal/qitian_ma_studbocconi_it/EuIaCYHnn4JCo0beGaSHhzIB6VZKP1xE2HbucCVb1qKVYw?e=h6VxoO

## Acknowledgments

This code is the result of teamwork at Bocconi University.


