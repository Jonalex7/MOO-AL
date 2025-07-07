import datetime
import pickle
import os
import argparse
import json
import time

import torch
import numpy as np
import yaml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

from limit_states import REGISTRY as ls_REGISTRY
from active_learning.active_learning import AcquisitionStrategy
from utils.data import isoprobabilistic_transform, custom_optimizer, parallel_predict, normalize_tensor

def main(config, name_exp):
    # getting args from config file
    casestudy = config['case_study'] # limit state to use
    al_strategy = config['al_strategy'] # active learning strategy
    al_batch = config['al_batch'] # number of samples to select at each iteration
    passive_samples = config['passive_samples'] # initial DoE with LHS
    total_samples = config['total_samples'] # max number of samples
    n_mcs_pool = config['n_mcs_pool'] # n_MonteCarlo pool of samples for learning
    n_mcs_pf = config['n_mcs_pf']  # n_MonteCarlo pool of samples for pf estimation
    seed_exp = config['seed'] # seed for experiment
    save_interval = config['save_interval']  # interval to save model

    # Loading limit state and ref. Pf
    lstate = ls_REGISTRY[casestudy]()
    Pf_ref = lstate.target_pf
    B_ref = - norm.ppf(Pf_ref)
    b_j = 0

    # results directory
    date_time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if al_strategy == 'moo':
        results_dir = f'results/{casestudy}/{al_strategy}_{config["moo_method"]}_{al_batch}_{name_exp}_{date_time_stamp}/'
    else:
        results_dir = f'results/{casestudy}/{al_strategy}_{al_batch}_{name_exp}_{date_time_stamp}/'

    store_model_dir = results_dir + 'model/'

    for dir_path in [results_dir, store_model_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Store the evolution of Pf, pareto metrics, and training samples
    results_file = {}
    pf_evol = []
    pareto_metrics = []

    # experiment seed for reproducibility
    if seed_exp is None:
        seed_exp = np.random.randint(0, 2**30 - 1)
    else:
        seed_exp=int(seed_exp)

    np.random.seed(seed_exp)
    torch.manual_seed(seed_exp)
    random_state = np.random.RandomState(seed_exp)
    config['seed'] = seed_exp  #saving seed

    # Store the config file as a json file
    with open(results_dir + 'config.json', 'w') as file_id:
        json.dump(config, file_id, indent=4)
    
    # Design of experiments
    x_train_norm, _ , y_train = lstate.get_doe(n_samples=passive_samples, method='lhs', random_state=random_state)

    iterations = int((total_samples-passive_samples)/al_batch) + 1 # number of iterations

    # Initializing the active learning strategy
    args_al = {
    'acquisition_strategy': al_strategy,
    'pareto_metrics': config['pareto_metrics'],  # If True, compute pareto front
    }
    # If moo strategy, add moo_method
    if al_strategy == 'moo':
        args_al['moo_method'] = config['moo_method']    # 'knee', 'compromised' 'reliability'
        # If moo_reliability strategy, add relevant parameters
        if args_al['moo_method'] == 'reliability': 
            args_al['N_it'] = config['N_it']  # Number of iterations to consider for moving average
            args_al['delta_P0'] = config['delta_p0'] # (0,1) threshold of relative difference at which gamma=0.5
            args_al['k_balance'] = config['k_balance'] # Positive constant controlling how quickly gamma transition from 0 to 1
    
    # Initialize the acquisition strategy
    strategy = AcquisitionStrategy(**args_al)

    start_time = time.time()
    print(f'Experiment settings: {config} \n')
    print(f'Reference Pf: {Pf_ref:.3E} \n')
    # Active learning loop
    for it in range(iterations + 1):
        
        print(f'Training size: {len(x_train_norm)} samples', end=" ")

        # Train the Gaussian Process model
        kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        model_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True, optimizer=custom_optimizer)
        model_gp.fit(x_train_norm, y_train)

        # Pf estimation with MCs
        x_mcs_pf = np.random.normal(0, 1, size=(int(n_mcs_pf), lstate.input_dim))
        mean_pf, _ = parallel_predict(model_gp, x_mcs_pf)
        Pf_model = (torch.sum(mean_pf < 0) / len(mean_pf)).item()
        pf_evol.append(Pf_model)

        # reliability index, B
        B_model = - norm.ppf(Pf_model)
        B_rel_diff = (B_model-B_ref)/B_ref

        # check beta stability
        b_stab = np.abs(B_model - b_j) / B_model   # relative difference with previous beta
        b_j = B_model  # Update b_j for the next iteration
        
        print(f'Pf_ref: {Pf_ref:.3E}, Pf_model: {Pf_model:.3E}, B_rel_diff: {B_rel_diff.item():.2%}, B_stab: {b_stab:.1%}')

        # Making predictions of mean and std for mc population 
        x_mc_pool = np.random.normal(0, 1, size=(int(n_mcs_pool), lstate.input_dim))
        # mean_prediction, std_prediction = model_gp.predict(x_mc_pool, return_std=True)
        mean_pred, std_pred = parallel_predict(model_gp, x_mc_pool)

        # arguments for sampling
        args_sampling = {'n_samples': 1, # Number of samples to select
                        'skip_indices': None} # Indices to skip in the pool
        
        # If the strategy is 'moo', we need to add the Pf estimate for reliability-based method
        if al_strategy == 'moo':
            if args_al['moo_method'] == 'reliability': 
                args_sampling['pf_estimate'] = Pf_model # Current Pf estimate for reliability method

        # Compute the indices to select based on the active learning strategy
        if args_al['pareto_metrics']:
            # If pareto metrics are enabled, we retrieve the pareto front and selected indices
            pareto, selected_indices = strategy.get_indices(
            mean_prediction=mean_pred,
            std_prediction=std_pred,
            **args_sampling
            )
            mean_pred_norm = normalize_tensor(torch.abs(mean_pred))
            std_pred_norm = normalize_tensor(std_pred)
            selected_objective_norm = torch.tensor([mean_pred_norm[selected_indices], std_pred_norm[selected_indices]])
            # Saving points for pareto metrics (full Pareto front, and selected sample)
            pareto_metrics.append((pareto.tolist(), selected_objective_norm.tolist()))
        else:
        # retrieve the selected indices without pareto metrics
            selected_indices = strategy.get_indices(
            mean_prediction=mean_pred,
            std_prediction=std_pred,
            **args_sampling
            )
        
        # Get training and target samples
        selected_samples_norm = x_mc_pool[selected_indices]

        # Converting to physical marginals and evaluating the model
        selected_samples = isoprobabilistic_transform(selected_samples_norm, lstate.standard_marginals, lstate.physical_marginals)
        selected_outputs = lstate.eval_lstate(selected_samples)

        # Update the training set
        selected_samples_torch = torch.tensor(selected_samples_norm)
        if selected_samples_torch.dim() == 1:
            selected_samples_torch = selected_samples_torch.unsqueeze(0)
        x_train_norm = torch.cat((x_train_norm, selected_samples_torch), 0)
        y_train = torch.cat((y_train, selected_outputs))

        # Saving results
        results_file['Pf_model'] = pf_evol

        if it % save_interval == 0:
            with open(results_dir + 'output.json', 'w') as file_id:
                        json.dump(results_file, file_id)

            # Save the model (pickle)
            with open(store_model_dir + 'gp_' + str(it) + '.pkl', 'wb') as file_id:
                pickle.dump(model_gp, file_id)

    # Saving final results
    results_file['Pf_model'] = pf_evol
    results_file['Pareto_metrics'] = pareto_metrics
    results_file['training_samples'] = x_train_norm.tolist(), y_train.tolist()  # training samples

    with open(results_dir + 'output.json', 'w') as file_id:
                    json.dump(results_file, file_id, indent=4)

    # Save the model (pickle)
    with open(store_model_dir + 'gp_' + "last" + '.pkl', 'wb') as file_id:
        pickle.dump(model_gp, file_id)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Active learning completed in: {(execution_time/60):.2f} mins")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-objective active learning for structural reliability')
    parser.add_argument('--config', type=str, nargs='?', action='store', default='default',
                        help='Configuration file name in config/ Def: default')
    parser.add_argument('--output', type=str, nargs='?', action='store', default='1',
                        help='Custom output file name Def: 1')
    
    # Parse known and unknown arguments
    args, unknown = parser.parse_known_args()
    name_exp = args.output # This can be used to define the experiment number

    # Loading experiment setting from config
    config_file = "config/" + args.config + ".yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Process unknown arguments to update the config
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip('-').replace('-', '_')
        value = unknown[i+1]

        if key in config:
            # Get the type from the config
            config_type = type(config[key])

            # Check if 'null' or similar was provided; convert to None
            if value.lower() in ('null', 'none'):
                value = None
            elif config[key] is not None:
                # Convert to the correct type
                if config_type == bool:
                    value = value.lower() in ('true', '1', 'yes')
                else:
                    value = config_type(value)
            else:
                # Convert to integer if possible
                value = int(value) if value.isdigit() else value

            config[key] = value
        else:
            print(f"Warning: Key '{key}' not found in config. Adding it as a new entry.")
            config[key] = value  # Add new key-value pair

    main(config=config, name_exp=name_exp)