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
from utils.data import isoprob_transform

#-----------------------------------------------------------------------------------------------
#arguments parser
parser = argparse.ArgumentParser(description='GP Regressor trained with batch active learning')

parser.add_argument('--ls', type=str, nargs='?', action='store', default='four_branch',
                    help='Specify target LS: four_branch, himmelblau, pushover_frame. Def: four_branch')

parser.add_argument('--acq_f', type=str, nargs='?', action='store', default='correlation',
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
n_mcs = 1e6  # n_MonteCarlo samples for pool

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
random_state = np.random.RandomState(seed_exp)

print(f'Target LS: {args.ls}, experiment seed: {seed_exp}')

#-----------------------------------------------------------------------------------------------
# Design of experiments
x_train_norm, x_train, y_train = lstate.get_doe(n_samples=doe, method='lhs', random_state=random_state)

active_learning = BatchActiveLearning(b_samples=args.acq_b)

# for it in range(iterations):
while len(x_train_norm) < budget+args.acq_b:
    
    print(f'Training size: {len(x_train_norm)} ')
    # Train the Gaussian Process model
    kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-07, 100000.0), nu=1.5)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(x_train_norm, y_train)

    # Make predictions over MC pool and estimate Pf
    print(f'Pf estimation over MC pool...', end=" ")
    x_mc_norm = np.random.uniform(0, 1, size=(int(n_mcs), lstate.input_dim))

    mean_prediction, std_prediction = gaussian_process.predict(x_mc_norm, return_std=True, return_cov=False)
    Pf_model = np.sum(mean_prediction < 0) / int(n_mcs)
    print(f'Pf model: {Pf_model.item():.3E} ')
    pf_evol.append(Pf_model)

    #Selection of batch of samples from MC pool
    selected_samples_norm = active_learning.get_u_function(x_mc_norm, mean_prediction, std_prediction)
    selected_samples = isoprob_transform(selected_samples_norm, lstate.marginals)
    selected_outputs = lstate.eval_lstate(selected_samples)

    # Update the training set
    x_train_norm = np.vstack([x_train_norm, selected_samples_norm])
    y_train = np.concatenate([y_train, selected_outputs])

print("Active learning completed")
