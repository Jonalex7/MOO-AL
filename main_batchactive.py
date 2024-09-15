import torch
import numpy as np
from datetime import datetime
import pickle
import os
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

import wandb
import time

from limit_states import REGISTRY as ls_REGISTRY
from active_learning.active_learning import BatchActiveLearning
from utils.data import min_max_distance, isoprobabilistic_transform

#-----------------------------------------------------------------------------------------------
#arguments parser
parser = argparse.ArgumentParser(description='GP Regressor trained with batch active learning')

parser.add_argument('--ls', type=str, nargs='?', action='store', default='four_branch',
                    help='Specify target LS: four_branch, himmelblau, pushover_frame. Def: four_branch')

parser.add_argument('--al_f', type=str, nargs='?', action='store', default='u_function',
                    help='Specify the acquisition function: correlation_det, correlation_eigen, correlation_entropy, u_function, random. Def: u_function')

parser.add_argument('--al_b', type=int, nargs='?', action='store', default=3,
                    help='Specify the batch sample size per iteration. Def: 3.')

parser.add_argument('--seed', type=int, nargs='?', action='store', default=None,
                    help='Specify the random seed. Def: Random')

parser.add_argument('--n_exp', type=int, nargs='?', action='store', default=1,
                    help='Specify the number of experiments to run. Def: 1')

args = parser.parse_args()

#-----------------------------------------------------------------------------------------------
# Loading limit state
lstate = ls_REGISTRY[args.ls]()
#-----------------------------------------------------------------------------------------------
# Loading experiment setting 
config={}
casestudy = config['case_study'] = args.ls
al_strategy = config['al_strategy'] = args.al_f
al_batch = config['al_batch'] = args.al_b
doe = config['doe'] = 10     # initial DoE with LHS
budget = config['budget'] = 200  # max number of samples
n_mcs_pool = config['n_mcs_pool'] = 1e6  # n_MonteCarlo pool of samples for learning
n_mcs_pf = config['n_mcs_pf']  = 1e6  # n_MonteCarlo pool of samples for pf estimation
n_exp = config['n_exp'] = args.n_exp
iterations = int((budget-doe)/args.al_b) + 1 #iteration to complete the available budget-doe

Pf_ref = lstate.target_pf
B_ref = - norm.ppf(Pf_ref)
b_j = 0

# Default seeds
seeds_exp= [391252418, 90523161, 375021598, 221860729, 45301975, 289396467, 698737664, 70359637, 800466323, 316421878]

#results directory
date_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'results/{casestudy}/{al_strategy}/{al_batch}/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print(f'Experiment settings: {config}')

for exp in range(args.n_exp):
    #results
    results_file = {}
    pf_evol = []

    #experiment seed for reproducibility
    if args.seed is not None:
        seed_exp = args.seed
    else:
        # seed_exp = np.random.randint(0, 2**30 - 1)
        seed_exp = seeds_exp[exp]

    np.random.seed(seed_exp)
    torch.manual_seed(seed_exp)
    random_state = np.random.RandomState(seed_exp)

    print(f'Experiment: {exp+1}/{args.n_exp}, Seed: {seed_exp}')
    config['seed'] = seed_exp
    results_file['config'] = config
    #-----------------------------------------------------------------------------------------------
    # log to wanb
    wandb.init(project='Batch_AL',
    name=f'{casestudy}_{al_strategy}_{al_batch}',
    config=config)
    #-----------------------------------------------------------------------------------------------
    # Design of experiments
    x_train_norm, _ , y_train = lstate.get_doe(n_samples=doe, method='lhs', random_state=random_state)

    # Loading active learning methods
    active_learning = BatchActiveLearning(n_active_samples= args.al_b)

    start_time = time.time()

    # Active learning loop
    for it in range(iterations + 1):
        # Find the minimum and maximum distance between training samples
        # min_distance, max_distance = min_max_distance(x_train_norm, x_train_norm)
        # kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(min_distance, max_distance), nu=1.5)
        
        print(f'Training size: {len(x_train_norm)} samples', end=" ")
        wandb.log({"train_size": len(x_train_norm)}, step=it)

        # Train the Gaussian Process model
        kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        model_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
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
            wandb.log({"Stop": len(x_train_norm)})

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

        # Converting to phyisical marginals and evaluating the model
        selected_samples = isoprobabilistic_transform(selected_samples_norm, lstate.standard_marginals, lstate.physical_marginals)
        selected_outputs = lstate.eval_lstate(selected_samples)

        # Update the training set
        x_train_norm = torch.cat((x_train_norm, torch.tensor(selected_samples_norm)), 0)
        y_train = torch.cat((y_train, selected_outputs))

        #saving partial results
        results_file['model'] = model_gp  
        results_file['Pf_model'] = pf_evol

        with open(results_dir+'exp_'+str(exp)+'_'+date_time_stamp+'.pkl', 'wb') as file_id:
                    pickle.dump(results_file, file_id)

    #saving final results
    results_file['model'] = model_gp  
    results_file['Pf_model'] = pf_evol
    results_file['training_samples'] = x_train_norm, y_train  #training samples

    with open(results_dir+'exp_'+str(exp)+'_'+date_time_stamp+'.pkl', 'wb') as file_id:
                    pickle.dump(results_file, file_id)
    end_time = time.time()
    execution_time = end_time - start_time

    wandb.log({"exc_time_mins": execution_time/60}, step=it)
    wandb.finish()
    print(f"Active learning completed in: {(execution_time/60):.2f} mins")
