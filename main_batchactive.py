import datetime
import pickle
import os
import argparse
import json
import time

import torch
import numpy as np
import yaml
import wandb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

from limit_states import REGISTRY as ls_REGISTRY
from active_learning.active_learning import BatchActiveLearning
from utils.data import isoprobabilistic_transform, custom_optimizer

def main(config, name_exp):
    # getting args from config file
    casestudy = config['case_study'] 
    al_strategy = config['al_strategy']  
    al_batch = config['al_batch'] 
    doe = config['doe'] # initial DoE with LHS
    budget = config['budget'] # max number of samples
    n_mcs_pool = config['n_mcs_pool'] # n_MonteCarlo pool of samples for learning
    n_mcs_pf = config['n_mcs_pf']  # n_MonteCarlo pool of samples for pf estimation
    seed_exp = config['seed'] # seed for experiment
    save_interval = config['save_interval'] 

    # Loading limit state and ref. Pf
    lstate = ls_REGISTRY[casestudy]()
    Pf_ref = lstate.target_pf
    B_ref = - norm.ppf(Pf_ref)
    b_j = 0

    # results directory
    date_time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = f'results/{casestudy}/{al_strategy}_{al_batch}_{name_exp}_{date_time_stamp}/'
    store_model_dir = results_dir + 'model/'

    for dir_path in [results_dir, store_model_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
    # results
    results_file = {}
    pf_evol = []
    stop_crit = []

    # experiment seed for reproducibility
    if seed_exp == "None":
        seed_exp = np.random.randint(0, 2**30 - 1)

    np.random.seed(seed_exp)
    torch.manual_seed(seed_exp)
    random_state = np.random.RandomState(seed_exp)
    config['seed'] = seed_exp  #saving seed

    # Store the config file as a json file
    with open(results_dir + 'config.json', 'w') as file_id:
        json.dump(config, file_id, indent=4)

    print(f'Experiment settings: {config}')
    ## log to wandb
    run_name = f'{name_exp}_{date_time_stamp}'
    wandb.init(project='Batch_AL', mode="offline", name=run_name, config=config)
    
    # Design of experiments
    x_train_norm, _ , y_train = lstate.get_doe(n_samples=doe, method='lhs', random_state=random_state)

    # Loading active learning methods
    active_learning = BatchActiveLearning(n_active_samples= al_batch)
    iterations = int((budget-doe)/al_batch) + 1 #iteration to complete the available budget-doe

    start_time = time.time()

    # Active learning loop
    for it in range(iterations + 1):
        
        print(f'Training size: {len(x_train_norm)} samples', end=" ")
        wandb.log({"train_size": len(x_train_norm)}, step=it)

        # Train the Gaussian Process model
        kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        model_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True, optimizer=custom_optimizer)
        model_gp.fit(x_train_norm, y_train)

        # Pf estimation with MCs
        x_mcs_pf = np.random.normal(0, 1, size=(int(n_mcs_pf), lstate.input_dim))
        mean_pf = model_gp.predict(x_mcs_pf, return_std=False)
        Pf_model = np.sum(mean_pf < 0) / len(mean_pf)
        pf_evol.append(Pf_model)

        # reliability index 
        B_model = - norm.ppf(Pf_model)
        B_rel_diff = (B_model-B_ref)/B_ref
        
        # check beta stability
        b_stab = np.abs(B_model - b_j) / B_model   #should be less than 0.005
        print(f'Pf_ref: {Pf_ref:.3E}, Pf_model: {Pf_model.item():.3E}, B_rel_diff: {B_rel_diff.item():.1%}, B_stab: {b_stab:.1%} \n')
        wandb.log({"pf_ref":Pf_ref, "pf_model": Pf_model, "b_rel": B_rel_diff, "b_stab": b_stab}, step=it)

        # Check if b_stab is smaller than 0.005
        if b_stab < 0.005:
            counter += 1
        else:
            counter = 0  # Reset counter if condition not met

        # Print 'stop' if b_stab is small for 3 consecutive iterations
        if counter >= 3:
            print(f'Stop at {len(x_train_norm)} samples!')
            stop_crit.append([len(x_train_norm), Pf_model, b_stab])

        b_j = B_model
        
        # Making predictions of mean and std for mc population 
        x_mc_pool = np.random.normal(0, 1, size=(int(n_mcs_pool), lstate.input_dim))
        mean_prediction, std_prediction = model_gp.predict(x_mc_pool, return_std=True)
        mean_pred = torch.tensor(mean_prediction)
        std_pred = torch.tensor(std_prediction)

        # Define the arguments for active learning
        args_al= {
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'x_mc_pool': x_mc_pool,
            'model': model_gp
        }
        # Select_indices method with the chosen active learning strategy
        selected_indices = active_learning.select_indices(al_strategy, **args_al)

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

        #saving partial results
        results_file['Pf_model'] = pf_evol

        if it % save_interval == 0:
            with open(results_dir + 'output.json', 'w') as file_id:
                        json.dump(results_file, file_id)

            # Save the model (pickle)
            with open(store_model_dir + 'gp_' + str(it) + '.pkl', 'wb') as file_id:
                pickle.dump(model_gp, file_id)

    #saving final results
    results_file['Pf_model'] = pf_evol
    results_file['b_stab'] = stop_crit
    results_file['training_samples'] = x_train_norm.tolist(), y_train.tolist()  #training samples

    with open(results_dir + 'output.json', 'w') as file_id:
                    json.dump(results_file, file_id, indent=4)

    # Save the model (pickle)
    with open(store_model_dir + 'gp_' + "last" + '.pkl', 'wb') as file_id:
        pickle.dump(model_gp, file_id)

    end_time = time.time()
    execution_time = end_time - start_time

    wandb.log({"exc_time_mins": execution_time/60}, step=it)
    wandb.finish()
    print(f"Active learning completed in: {(execution_time/60):.2f} mins")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GP Regressor trained with batch active learning')
    parser.add_argument('--config', type=str, nargs='?', action='store', default='default',
                        help='Configuration file name in config/ Def: default')
    parser.add_argument('--output', type=str, nargs='?', action='store', default='1',
                        help='Custom output file name Def: 1')
    args = parser.parse_args()

    name_exp = args.output # This can be used to define the experiment number

    # Loading experiment setting from config
    config_file = "config/" + args.config + ".yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    
    main(config=config, name_exp=name_exp)