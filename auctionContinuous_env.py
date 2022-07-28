import torch

class AuctionContEnv:
    def __init__(self, device, n_bidders=5, clickRates_ls=(20, 10, 5, 2, 0), valuations_ls=(5, 4, 3, 2, 1), bidMin=.2, bidMax=5, seed=None):
        assert len(clickRates_ls) == n_bidders # valid input length for click rates
        assert len(valuations_ls) == n_bidders # valid input length for valuations
        assert bidMin > 0 # minimal bids acceptable should be greater than 0
        assert bidMax > bidMin # maxximal bids acceptable hsould be greater than minimal bid acceptable
        
        self.device = device # GPU device
        self.n_bidders = n_bidders # number of bidder
        self.clickRates_ls = torch.tensor(  # click rate of bidders
            clickRates_ls, dtype=torch.float32).reshape(-1, ).to(self.device)
        self.valuations_ls = torch.tensor(
            valuations_ls, dtype=torch.float32).reshape(-1, ).to(self.device)
        self.bidMin = bidMin # min bids acceptable
        self.bidMax = bidMax # max bids acceptable
        self.clickRates_ls = self.clickRates_ls.sort(descending=True).values # sort click rates
        
        
        assert (self.clickRates_ls < 0).sum() == 0 # click rates should all be non-negative
        assert (self.valuations_ls < 0).sum() == 0 # valuations should all be non-negative

        
        self.reset(seed=seed)

    def __sort(self, actions_ls):
        """
        clickRates_ls is sorted
        In order to compute price, bids need to be sorted
        sort actions takes a list and return 3 lists:
        actions_sorted_ls: sorted actions
        actions_argsort_idx: arg sort
        actions_iargsort_idx: inverse arg sort
        """
        actions_sorted_ls, actions_argsort_idx = actions_ls.sort(
            descending=True)
        actions_iargsort_idx = torch.zeros(
            size=(self.n_bidders,), dtype=torch.int64)
        actions_iargsort_idx[actions_argsort_idx] = torch.arange(
            self.n_bidders)
        return actions_sorted_ls, actions_argsort_idx, actions_iargsort_idx

    def step(self, actions_ls):
        
        """
        accept actions_ls, which is the list of actions taken by the bidder
        update state, which is the list of actions taken by the bidder
        return actions_ls, reward and number of steps taken
        
        Pay attention to the user of argsort and iargsort
        Any value list that corresponds to the original actions, indexed on argsort will result in an order same as sorted actions. That is, the first value corresponds to the highest bidder.
        Any value list that corresponds to the sorted actions, indexed on iargsort will result in an order same as original actions. That is, the first value corresponds to the first bidder.
        # """
        assert actions_ls.type().split(".")[-1] == "FloatTensor"
        assert actions_ls.dim() == 1
        assert len(actions_ls) == self.n_bidders
        assert actions_ls.min() >= self.bidMin
        assert actions_ls.max() <= self.bidMax
        

        actions_sorted_ls, actions_argsort_idx, actions_iargsort_idx = self.__sort(
            actions_ls)

        actions_sorted_diff_ls = actions_sorted_ls.diff()

        # Rare event. Same bids occur. Apply a small random pertubation. 
        if 0 in actions_sorted_diff_ls:
            zero_idx = (actions_sorted_diff_ls == 0).nonzero().reshape(-1, )
            actions_sorted_ls[zero_idx] += (
                torch.rand(size=(len(zero_idx),)) - .5).to(self.device) * 1E-10

        actions_ls = actions_sorted_ls[actions_iargsort_idx]
        
        actions_sorted_ls, actions_argsort_idx, actions_iargsort_idx = self.__sort(
            actions_ls)
        
        # Each bidder's price is set to be the lower neigbouring bid. The price of the last bid is 0.
        price_ls = actions_ls[actions_argsort_idx].roll(shifts=-1)
        price_ls[-1] = 0
        
        # The reward is calculated as 
        rewards_ls = (self.valuations_ls -
                  price_ls[actions_iargsort_idx]) * self.clickRates_ls[actions_iargsort_idx]

        self.n_steps += 1
        self.states = actions_ls
        
        actions_ls = actions_ls.to(self.device)
        rewards_ls = rewards_ls.to(self.device)
        return actions_ls, rewards_ls, self.n_steps

    def reset(self, seed):
        """
        Initialize the first state radomly.
        """
        if seed:
            assert isinstance(seed, int)
            torch.manual_seed(seed)
        self.states = torch.rand(size=(self.n_bidders,)) * (self.bidMax-self.bidMin) + self.bidMin
        self.states = self.states.to(self.device)
        self.n_steps = 0