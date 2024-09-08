import torch
import numpy as np
from datetime import datetime
import pickle
import os
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# import wandb
import time

from limit_states import REGISTRY as ls_REGISTRY
from active_learning.active_learning import BatchActiveLearning
from utils.data import isoprob_transform, min_max_distance

#-----------------------------------------------------------------------------------------------
#arguments parser
parser = argparse.ArgumentParser(description='GP Regressor trained with batch active learning')

parser.add_argument('--ls', type=str, nargs='?', action='store', default='four_branch',
                    help='Specify target LS: four_branch, himmelblau, pushover_frame. Def: four_branch')

parser.add_argument('--acq_f', type=str, nargs='?', action='store', default='u_function',
                    help='Specify the acquisition function: correlation_det, correlation_eigen, u_function, random. Def: u_function')

parser.add_argument('--acq_b', type=int, nargs='?', action='store', default=3,
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
doe = config['doe'] = 10     # initial DoE with LHS
budget = config['budget'] = 100  # max number of samples
n_mcs_pool = config['n_mcs_pool'] = 1e5  # n_MonteCarlo pool of samples for learning
n_mcs_pf = config['n_mcs_pf']  = 1e6  # n_MonteCarlo pool of samples for pf estimation
iterations = int((budget-doe)/args.acq_b) + 1 #iteration to complete the available budget-doe

#results directory
date_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'results/{args.ls}/{args.acq_f}/{args.acq_b}/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print(f'Target LS: {args.ls}')

for exp in range(args.n_exp):
    #results
    results_file = {}
    pf_evol = []

    #experiment seed for reproducibility
    if args.seed is not None:
        seed_exp = args.seed
    else:
        seed_exp = np.random.randint(0, 2**30 - 1)

    np.random.seed(seed_exp)
    torch.manual_seed(seed_exp)
    random_state = np.random.RandomState(seed_exp)

    print(f'Experiment: {exp+1}/{args.n_exp}, Seed: {seed_exp}')
    config['seed'] = seed_exp
    results_file['config'] = config  
    #-----------------------------------------------------------------------------------------------
    # Design of experiments
    x_train_norm, _ , y_train = lstate.get_doe(n_samples=doe, method='lhs', random_state=random_state)
    x_train_norm = torch.tensor(x_train_norm)
    y_train = torch.tensor(y_train)

    # Loading active learning methods
    active_learning = BatchActiveLearning(n_active_samples= args.acq_b)

    start_time = time.time()

    for it in range(iterations):
        # Find the minimum and maximum distance between training samples
        # min_distance, _ = min_max_distance(x_train_norm, x_train_norm)
        # kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(min_distance, 1000000), nu=1.5)
        
        print(f'Training size: {len(x_train_norm)} samples', end=" ")
        # Train the Gaussian Process model
        kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        model_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        model_gp.fit(x_train_norm, y_train)

        # Pf estimation with MCs
        x_mcs_pf = np.random.uniform(0, 1, size=(int(n_mcs_pf), lstate.input_dim))
        mean_pf = model_gp.predict(x_mcs_pf, return_std=False)
        Pf_model = np.sum(mean_pf < 0) / len(mean_pf)
        print(f'Pf ref: {lstate.target:.3E}, Pf model: {Pf_model.item():.3E} \n')
        pf_evol.append(Pf_model)

        # Making predictions of mean and std for mc population 
        x_mc_pool = np.random.uniform(0, 1, size=(int(n_mcs_pool), lstate.input_dim))
        mean_prediction, std_prediction = model_gp.predict(x_mc_pool, return_std=True)
        mean_pred = torch.tensor(mean_prediction)
        std_pred = torch.tensor(std_prediction)
            
        if args.acq_f == 'u_function':
            selected_indices = active_learning.get_u_function(mean_pred, std_pred)

        elif args.acq_f == 'correlation_det':
            selected_indices = active_learning.get_correlation_det(x_mc_pool, model_gp, mean_pred, std_pred)

        elif args.acq_f == 'correlation_eigen':
            selected_indices = active_learning.get_correlation_eigen(x_mc_pool, model_gp, mean_pred, std_pred)
        
        elif args.acq_f == 'random':
            selected_indices = active_learning.get_random(x_mc_pool)

        # Get training and target samples
        selected_samples_norm = x_mc_pool[selected_indices]

        selected_samples = isoprob_transform(selected_samples_norm, lstate.marginals)
        selected_outputs = lstate.eval_lstate(selected_samples)

        # Update the training set
        x_train_norm = torch.cat((x_train_norm, torch.tensor(selected_samples_norm)), 0)
        y_train = torch.cat((y_train, torch.tensor(selected_outputs)))

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
    print(f"Active learning completed in: {(execution_time/60):.2f} mins")
