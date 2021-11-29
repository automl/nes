import os
import copy
import random
import re
import json

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np

from hpbandster.core.base_config_generator import base_config_generator
from hpbandster.core.result import Run
from hpbandster.core.result import logged_results_to_HBS_result as res_loader

from nes.ensemble_selection.utils import model_seeds
from nes.ensemble_selection.esas import registry as esas_registry
from nes.ensemble_selection.esas import run_esa
from nes.ensemble_selection.containers import load_baselearner


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


class RegularizedEvolutionSampler(base_config_generator):
    def __init__(self, working_dir, configspace, population_size=50,
                 pop_sample_size=10, scheme='nes_re', esa='beam_search',
                 severity_list=[0, 5], warmstart_dir=None, **kwargs):
        """Base class for running the NES-RE computation.

        Args:
            working_dir     (str): directory where results are written
            configspace     (ConfigSpace.ConfigurationSpace): valid representation
                of the search space
            population_size (int): total population size in NES-RE.
            pop_sample_size (int): ensemble size used in forward select
            scheme          (str): scheme name
            esa             (str): ensemble selection algorithm, e.g. ForwardSelect
            warmstart_dir   (str): directory where previous results are stored.
                Used only when it is not None
            severity_list   (list): list of severities where NES-RE uniformly
                samples from during the NES-RE search.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.configspace = configspace
        self.working_dir = working_dir
        self.population_size = population_size
        self.pop_sample_size = pop_sample_size
        self.esa = esa
        self.scheme = scheme
        self.severity_list = severity_list

        self.population = list()
        self.init_compute_queue = list()
        self.history = list()

        if warmstart_dir is not None:
            # load the logged results from hpbandster
            previous_results = res_loader(warmstart_dir)
            self.history_len = len(previous_results.get_all_runs())
            # number of epochs in this case
            budget = previous_results.HB_config['budgets'][0]

            # load the last population ids
            with open(os.path.join(warmstart_dir, 'history.json'), 'r') as f:
                self.logger.info(f'>>>>>>> Loading history from {warmstart_dir}')
                self.history.extend(json.load(f))
                last_population = self.history[-1]

            self.logger.info(f'>>>>>>> Warmstarting population from {warmstart_dir}')
            for model_id in last_population:
                config_data = previous_results.get_all_runs()[model_id]
                # this should be (config_id, 0, 0)
                hbs_config_id = config_data.config_id
                # this extracts the architecture corresponding to this config id
                config_dict = previous_results.get_id2config_mapping()[hbs_config_id]['config']

                run = Run(config_id=model_id, budget=budget,
                          loss=config_data.loss, info={'config': config_dict,
                                           'dest_dir': config_data.info['dest_dir']},
                          time_stamps={'submitted': 0, 'started': 0, 'finished': 0},
                          error_logs=0)
                self.population.append(run)
        else:
            self.create_initial_population()


        hps = self.configspace.get_hyperparameters()
        for h in hps:
            if hasattr(h, 'lower'):
                raise RuntimeError(
                    'This version of RE only works with categorical '
                    'parameters (encoding the arch!' % (h.name))


    def create_initial_population(self):
        # Create initial population by uniformly sampling self.population_size
        # architectures at random
        for i in range(self.population_size):
            sample_config_dict = self.configspace.sample_configuration().get_dictionary()
            self.init_compute_queue.append(sample_config_dict)


    def select_new_parent(self, budget):
        assert len(self.init_compute_queue) == 0, 'A new parent should only be chosen once the compute queue is empty.'

        # wrap the trained architectures in the current population with the
        # Baselearner class
        population_baselearners = {
            x.config_id: load_baselearner(model_id=model_seeds(x.config_id,
                                                               x.config_id,
                                                               self.scheme),
                                          load_nn_module=False,
                                          working_dir=x.info['dest_dir']) for x in self.population
        }

        # randomly choose one severity from self.severity_list to evaluate the
        # baselearners when running ForwardSelect
        severity = random.choice(self.severity_list)

        # select the ensemple of size self.pop_sample_size based on the ensemble
        # selection algorithm (esa), i.e. in this case ForwardSelect
        pop_sample = run_esa(M=self.pop_sample_size,
                             population=population_baselearners,
                             esa=esas_registry[self.esa],
                             val_severity=severity)

        self.logger.info('ESA: {}'.format(pop_sample))

        # append current population to history and save that
        self.history.append(list(population_baselearners.keys()))
        with open(os.path.join(self.working_dir, 'history.json'), 'w') as g:
            json.dump(self.history, g)

        # select one random model in pop_sample (selected ensemble from esa)
        parent_id = random.choice(pop_sample)
        parent_baselearner = list(filter(lambda x: x.config_id == parent_id,
                                         self.population))
        assert len(parent_baselearner) == 1
        parent_baselearner = parent_baselearner[0]

        # wrap the parent_baselearner dict with a ConfigSpace.Configuration
        # object (needed for hpbandster and the mutations)
        parent_config = ConfigSpace.Configuration(self.configspace,
                                                  parent_baselearner.info['config'])
        # mutate the parent configuration and return the child.
        # self.configspace is the search space object
        mutated_config_dict = self.mutate_arch(self.configspace,
                                               parent_config).get_dictionary()

        # Evaluate the mutated parent next
        return mutated_config_dict

    def get_config(self, budget):
        """Function to sample a new architecture configuration.

        Args:
            budget (float): the budget for which this configuration is scheduled

        Returns:
            Configuration: should return a valid architecture configuration
            dict: a dictionary with other info

        """
        self.logger.info('>>>>>>> Population size: {}'.format(len(self.population)))

        if len(self.init_compute_queue) > 0:
            # Evaluate the initial population first
            sample = self.init_compute_queue.pop()
        else:
            # Select a new sample from the population with replacement as 
            # done by regularized evolution:
            # https://github.com/google-research/google-research/blob/master/evolution/
            # regularized_evolution_algorithm/regularized_evolution.ipynb
            sample = self.select_new_parent(budget)

        info_dict = {}
        return sample, info_dict


    def new_result(self, job, update_model=True):
        """Function to register finished runs and adjust the population.

            Every time a run has finished, this function should be called
            to register it with the result logger. If overwritten, make
            sure to call this method from the base class to ensure proper
            logging.

        Args:
            job (hpbandster.distributed.dispatcher.Job): contains all the info
                about the run
            update_model (bool): hpbandster internal things

        Returns:
            None
        """
        self.logger.info('>>>>>>> Trying to register results')

        if job.result is None:
            # One could skip crashed results, but we decided to
            # assign a +inf loss and count them as bad configurations
            loss = np.inf
        else:
            loss = job.result['loss']

        budget = job.kwargs["budget"]
        config_dict = job.kwargs["config"]
        # matches the arch_id an seed_id
        model_id = job.result['info']['model_id']

        # create a Run object so hpbandster can log the results automatically
        run = Run(config_id=model_id, budget=budget,
                  loss=loss, info={'config': config_dict,
                                   'dest_dir': job.result['info']['dest_dir']},
                  time_stamps={'submitted': 0, 'started': 0, 'finished': 0},
                  error_logs=0)

        # remove from the population the oldest individual
        if len(self.population) >= self.population_size:
            self.population.pop(0)

        # Record the value of the individual
        self.population.append(run)


    def mutate_arch(self, cs, parent_config_instance):
        """Function that applies the mutations to the architecture at hand.

        Args:
            cs (ConfigSpace.ConfigurationSpace): search space ConfigSpace object
            parent_config_instance (ConfigSpace.Configuration): parent
                architecture

        Returns:
            ConfigSpace.Configuration: child architecture
        """
        # Select which cell type to mutate
        cell_type = np.random.choice(['normal', 'reduce'])
        # Choose one of the three mutations
        mutation = np.random.choice(['identity', 'hidden_state_mutation',
                                     'op_mutation'])

        if mutation == 'identity':
            return copy.deepcopy(parent_config_instance)
        elif mutation == 'hidden_state_mutation':
            # Create child architecture with modified link
            child_arch_dict = copy.deepcopy(
                parent_config_instance.get_dictionary()
            )
            # Get all active hyperparameters which are related to 
            # the adjacency matrix of the cells
            hidden_states = list(
                filter(re.compile('.*inputs_node_{}*.'.format(cell_type)).match,
                       cs.get_active_hyperparameters(parent_config_instance)))

            # Select one hidden state to modify
            selected_hidden_state = cs.get_hyperparameter(
                str(np.random.choice(hidden_states))
            )

            # Choose the parent to change.
            current_parents = [int(parent) for parent in
                               child_arch_dict[selected_hidden_state.name].split('_')]
            removed_parent = np.random.choice(current_parents)
            current_parents.remove(removed_parent)
            [remaining_parent] = current_parents

            # Determine the active intermediate nodes in the cell
            active_intermediate_nodes = []
            for state in hidden_states:
                active_intermediate_nodes.extend(
                    [int(intermediate_node) for intermediate_node in
                     parent_config_instance[cs.get_hyperparameter(state).name].split('_')]
                )

            # Which parent combinations contain the parent_to_stay 
            # and which operation edge is affected?
            node_parent_to_edge_num = \
                    lambda parent, node: parent + sum(np.arange(2, node + 1)) - node

            # Remove the previous edge
            selected_node = int(selected_hidden_state.name[-1])
            deleted_edge = node_parent_to_edge_num(removed_parent,
                                                   selected_node)
            op_for_new_edge = child_arch_dict[
                'NetworkSelectorDatasetInfo:darts:edge_{}_{}'.format(cell_type,
                                                                     deleted_edge)]
            child_arch_dict = \
                removekey(child_arch_dict,
                          'NetworkSelectorDatasetInfo:darts:edge_{}_{}'.format(cell_type,
                                                                               deleted_edge))

            # Select the new parent from active intermediate nodes
            possible_parents = [i for i in
                                np.sort(np.unique(active_intermediate_nodes))
                                if i < selected_node]
            # Remove current parent
            possible_parents.remove(remaining_parent)
            new_parent = np.random.choice(possible_parents)
            new_parents = '_'.join([str(elem) for elem in np.sort([new_parent,
                                                                   remaining_parent])])

            # Add new edge
            new_edge = node_parent_to_edge_num(new_parent, selected_node)
            child_arch_dict['NetworkSelectorDatasetInfo:darts:edge_{}_{}'.format(cell_type,
                                                                                 new_edge)] = op_for_new_edge

            # Add new parents
            child_arch_dict[selected_hidden_state.name] = new_parents

            child_config_instance = ConfigSpace.Configuration(cs,
                                                              values=child_arch_dict)
            return child_config_instance

        else:
            # op mutation
            # Get all active hyperparameters which are related 
            # to the operations chosen in the cell
            hidden_state = list(
                filter(re.compile('.*edge_{}*.'.format(cell_type)).match,
                       cs.get_active_hyperparameters(parent_config_instance)))
            selected_hidden_state = cs.get_hyperparameter(
                str(np.random.choice(hidden_state))
            )
            choices = list(selected_hidden_state.choices)

            # Drop current value from the list of choices
            choices.remove(parent_config_instance[selected_hidden_state.name])

            # Create child architecture with modified link
            child_arch_dict = copy.deepcopy(
                parent_config_instance.get_dictionary()
            )

            # Modify the selected link
            child_arch_dict[selected_hidden_state.name] = str(
                np.random.choice(choices)
            )
            child_config_instance = ConfigSpace.Configuration(
                cs, values=child_arch_dict
            )
            return child_config_instance
