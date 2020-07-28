import numpy as np

from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving

from .re_gen import RegularizedEvolution as CG_RE


class RegularizedEvolution(Master):
    def __init__(self, working_directory, configspace=None, eta=3, min_budget=1, max_budget=1,
                 population_size=50, pop_sample_size=10, scheme='nes_re',
                 esa='beam_search', warmstart_dir=None, severity_list=[0, 5],
                 **kwargs):
        """Master node implementation that runs NES-RE.

        Args:
            working_directory (str): directory where results are written
            configspace       (ConfigSpace.ConfigurationSpace): valid representation
                of the search space
            eta               (int): multiplicative factor (not used ever in
                NES). Needed by hpbandster
            min_budget        (int): Do not change the default. Always set to 1.
            max_budget        (int): Do not change the default. Always set to 1.
            population_size   (int): total population size in NES-RE.
            pop_sample_size   (int): ensemble size used in forward select
            scheme            (str): scheme name
            esa               (str): ensemble selection algorithm, e.g. ForwardSelect
            warmstart_dir     (str): directory where previous results are stored.
                Used only when it is not None
            severity_list     (list): list of severities where NES-RE uniformly
                samples from during the NES-RE search.

        Returns:
            None
        """
        # TODO: Proper check for ConfigSpace object!
        if configspace is None:
            raise ValueError("You have to provide a valid ConfigSpace object")

        cg = CG_RE(working_dir=working_directory,
                   configspace=configspace,
                   population_size=population_size,
                   pop_sample_size=pop_sample_size,
                   scheme=scheme,
                   esa=esa,
                   severity_list=severity_list,
                   warmstart_dir=warmstart_dir)

        super().__init__(config_generator=cg, **kwargs)

        ############     below some hpbandster related stuff is done     ############
        ############               not important for NES                 ############
        ############ necessary in order to be compatible with hpbandster ############

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = max_budget
        self.max_budget = max_budget

        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0,
                                                               self.max_SH_iter))

        # max total budget for one iteration
        self.budget_per_iteration = sum([b * self.eta**i for i, b in
                                         enumerate(self.budgets[::-1])])

        self.config.update({
            'population_size': population_size,
            'pop_sample_size': pop_sample_size
        })


    def get_next_iteration(self, iteration, iteration_kwargs={}):
        """Returns a SH iteration with only evaluations on the biggest budget.
            Not related to NES. It is only to be compatible with hpbandster.

        Args:
            iteration        (int): the index of the iteration to be instantiated
            iteration_kwargs (dict): kwargs to be added to the instantiation of
                each iteration

        Returns:
            SuccessiveHalving: the SuccessiveHalving iteration with the
                corresponding number of configurations
        """

        budgets = [self.max_budget]
        ns = [self.budget_per_iteration // self.max_budget]

        return (SuccessiveHalving(HPB_iter=iteration, num_configs=ns, budgets=budgets,
                                  config_sampler=self.config_generator.get_config,
                                  **iteration_kwargs))

