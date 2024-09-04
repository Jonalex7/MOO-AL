import torch
import numpy as np
# import yaml
from datetime import datetime
import pickle
import os
# import wandb
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from limit_states import REGISTRY as ls_REGISTRY
from active_learning.active_learning import BatchActiveLearning
from utils.data import isoprob_transform, min_max_distance

#-----------------------------------------------------------------------------------------------
#arguments parser
parser = argparse.ArgumentParser(description='GP Regressor trained with batch active learning')

parser.add_argument('--ls', type=str, nargs='?', action='store', default='four_branch',
                    help='Specify target LS: four_branch, himmelblau, pushover_frame. Def: four_branch')

parser.add_argument('--acq_f', type=str, nargs='?', action='store', default='u_function',
                    help='Specify the acquisition function: correlation, u_function. Def: correlation')

parser.add_argument('--acq_b', type=int, nargs='?', action='store', default=3,
                    help='Specify the batch sample size per iteration. Def: 3.')

parser.add_argument('--seed', type=int, nargs='?', action='store', default=None,
                    help='Specify the random seed. Def: Random')

args = parser.parse_args()

#-----------------------------------------------------------------------------------------------
# Loading limit state
lstate = ls_REGISTRY[args.ls]()

#-----------------------------------------------------------------------------------------------
# Loading experiment setting 
doe =  10     # initial DoE with LHS
budget = 100  # max number of samples
n_mcs_pool = 1e5  # n_MonteCarlo pool of samples for learning
n_mcs_pf = 1e6  # n_MonteCarlo pool of samples for pf estimation
iterations = int((budget-doe)/args.acq_b) + 1 #iteration to complete the available budget-doe

# # Create list of indices for learning pool
# indices = torch.arange(int(n_mcs_pool))

#results directory
date_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'results/{args.ls}/{args.acq_f}/{args.acq_b}/'
results_file = {}
pf_evol = []

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

#experiment seed for reproducibility
if args.seed is not None:
    seed_exp = args.seed
else:
    seed_exp = np.random.randint(0, 2**30 - 1)

np.random.seed(seed_exp)
torch.manual_seed(seed_exp)
random_state = np.random.RandomState(seed_exp)

print(f'Target LS: {args.ls}, experiment seed: {seed_exp}')

#-----------------------------------------------------------------------------------------------
# Design of experiments
x_train_norm, _ , y_train = lstate.get_doe(n_samples=doe, method='lhs', random_state=random_state)
x_train_norm = torch.tensor(x_train_norm)
y_train = torch.tensor(y_train)

# Loading active learning methods
active_learning = BatchActiveLearning(n_active_samples= args.acq_b)

for it in range(iterations):
# while len(x_train_norm) < budget+args.acq_b:

    # Find the minimum and maximum distance between training samples
    # min_distance, max_distance = min_max_distance(x_train_norm, x_train_norm)
    # kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(min_distance, max_distance), nu=1.5)
    
    print(f'Training size: {len(x_train_norm)} ')
    # Train the Gaussian Process model
    kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    model_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    model_gp.fit(x_train_norm, y_train)

    # Pf estimation with MCs
    # print(f'Pf estimation over MC pool...', end=" ")
    x_mcs_pf = np.random.uniform(0, 1, size=(int(n_mcs_pf), lstate.input_dim))
    mean_pf = model_gp.predict(x_mcs_pf, return_std=False)
    Pf_model = np.sum(mean_pf < 0) / len(mean_pf)
    print(f'Pf ref: {lstate.target:.3E}, Pf model: {Pf_model.item():.3E} \n')
    # pf_evol.append(Pf_model)

    # Making predictions of mean and std for mc population 
    x_mc_pool = np.random.uniform(0, 1, size=(int(n_mcs_pool), lstate.input_dim))
    mean_prediction, std_prediction = model_gp.predict(x_mc_pool, return_std=True)
    mean_pred = torch.tensor(mean_prediction)
    std_pred = torch.tensor(std_prediction)
        
    if args.acq_f == 'u_function':
        selected_indices = active_learning.get_u_function(mean_pred, std_pred)

    elif args.acq_f == 'correlation':
        selected_indices = active_learning.get_correlation(x_mc_pool, model_gp, mean_pred, std_pred)

    # Get training and target samples
    selected_samples_norm = x_mc_pool[selected_indices]

    selected_samples = isoprob_transform(selected_samples_norm, lstate.marginals)
    selected_outputs = lstate.eval_lstate(selected_samples)

    # Update the training set
    x_train_norm = torch.cat((x_train_norm, torch.tensor(selected_samples_norm)), 0)
    y_train = torch.cat((y_train, torch.tensor(selected_outputs)))

    # print(x_train_norm)

print("Active learning completed")
