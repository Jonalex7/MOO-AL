import datetime
import os
import argparse
import json
import time

import numpy as np
import yaml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
import wandb

from limit_states import REGISTRY as ls_REGISTRY
from active_learning.active_learning import AcquisitionStrategy
from utils.data import isoprobabilistic_transform, custom_optimizer, normalize_array, parallel_predict

def make_base_kernel(input_dim):
    length_init = np.full(input_dim, 1.0, dtype=np.float64)
    kernel = ConstantKernel(1.0, (1e-5, 1e5)) * Matern(
        length_scale=length_init,
        length_scale_bounds=(1e-5, 1e5),
        nu=2.5,
    )
    return kernel

def is_bad_fit(current_lml, prev_lml, lml_drop_tol=50.0, abs_lml_low=-100.0):
    """
    Consider a fit 'bad' if:
      - LML drops a lot compared to the previous good model, OR
      - LML is absolutely very low.
    """
    too_low = current_lml < abs_lml_low
    if prev_lml is None:
        big_drop = False
    else:
        big_drop = current_lml < (prev_lml - lml_drop_tol)
    return too_low or big_drop


def main(config, name_exp):
    wandb_mode = "online" if config.get("wandb_online", False) else "offline"
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
    config['name_exp'] = name_exp

    # Loading limit state and ref. Pf
    lstate = ls_REGISTRY[casestudy]()
    Pf_ref = lstate.target_pf
    B_ref = - norm.ppf(Pf_ref)

    # results directory
    date_time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if al_strategy == 'moo':
        results_dir = f'results/{casestudy}/{al_strategy}_{config["moo_method"]}_{al_batch}_{name_exp}_{date_time_stamp}/'
    else:
        results_dir = f'results/{casestudy}/{al_strategy}_{al_batch}_{name_exp}_{date_time_stamp}/'

    # store_model_dir = results_dir + 'model/'

    for dir_path in [results_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Store the evolution of Pf, pareto metrics, and training samples
    results_file = {}
    pf_evol = []
    pareto_metrics = []
    lml_evol = []

    # experiment seed for reproducibility
    if seed_exp is None:
        seed_exp = np.random.randint(0, 2**30 - 1)
    else:
        seed_exp=int(seed_exp)

    np.random.seed(seed_exp)
    random_state = np.random.RandomState(seed_exp)
    config['seed'] = seed_exp  #saving seed

    # Store the config file as a json file
    with open(results_dir + 'config.json', 'w') as file_id:
        json.dump(config, file_id, indent=4)
    
    # Design of experiments
    x_train_norm, _ , y_train = lstate.get_doe(n_samples=passive_samples, method='lhs', random_state=random_state)
    x_train_norm = np.asarray(x_train_norm, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

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
            args_al['input_dim'] = lstate.input_dim
    
        if args_al['moo_method'] == "eps_greedy" or args_al['moo_method'] == "eps_lw":
            args_al['eps_start'] = config['eps_start']  # Initial epsilon value
            args_al['eps_end'] = config['eps_end']      # Final epsilon value
            args_al['eps_T'] = config['eps_T']          # Number of iterations to decay over

    if al_strategy == "portfolio":
        args_al['portfolio_lambda'] = config['portfolio_lambda']  # Hedge balance (λ)
        args_al['portfolio_delta'] = config['portfolio_delta']    # Memory factor (δ)

    if al_strategy == "eier":
        if al_batch != 1:
            raise ValueError("EIER strategy currently supports al_batch=1 only.")
        args_al['batch_size_acq'] = config['batch_size_acq']
        args_al['n_z_mc'] = config['n_z_mc']
        args_al['jitter_stddev'] = config['obs_stddev']
        args_al['local_mis_topk'] = config['local_mis_topk']
        args_al['debug_acq'] = config.get('debug_acq', False)
        
    # Initialize the acquisition strategy
    strategy = AcquisitionStrategy(**args_al)
    
    # log experiment with wandb
    run_name = f'{name_exp}_{date_time_stamp}'
    wandb.init(project="MOO_AL", name=run_name, config=config, 
               mode=wandb_mode, settings=wandb.Settings(start_method="thread"))
    
    # Before the AL loop
    kernel_prev = None     # last good kernel
    lml_prev = None        # LML of last good model

    start_time = time.time()
    print(f'Experiment settings: {config} \n')
    print(f'Reference Pf: {Pf_ref:.3E} \n')
    # Active learning loop
    for it in range(iterations + 1):
        
        print(f'Training samples: {len(x_train_norm)} |', end=" ")
        wandb.log({"train_size": len(x_train_norm)}, step=it)

        # --- 1) Choose initialization kernel ---
        if kernel_prev is None:
            # first iteration: base kernel
            init_kernel = make_base_kernel(lstate.input_dim)
        else:
            # warm-start from last good kernel
            init_kernel = kernel_prev

        gp_alpha = 1e-8
        if al_strategy == "eier":
            y_scale = float(np.std(y_train))
            y_scale = max(y_scale, 1e-12)
            gp_alpha = float(config['obs_stddev'] / y_scale) ** 2

        # Train the Gaussian Process model
        model_gp = GaussianProcessRegressor(
            kernel=init_kernel,
            n_restarts_optimizer=0,      # refine around warm-start
            normalize_y=True,
            optimizer=custom_optimizer,
            alpha=gp_alpha
        )
        model_gp.fit(x_train_norm, y_train)
        lml = model_gp.log_marginal_likelihood_value_
        
        if is_bad_fit(lml, lml_prev, lml_drop_tol=50.0, abs_lml_low=-100.0):
            # This fit looks suspicious -> try a fresh base kernel with restarts
            base_kernel = make_base_kernel(lstate.input_dim)
            model_gp_fresh = GaussianProcessRegressor(
                kernel=base_kernel,
                n_restarts_optimizer=9,   # full search from scratch
                normalize_y=True,
                optimizer=custom_optimizer,
                alpha=gp_alpha
            )
            model_gp_fresh.fit(x_train_norm, y_train)
            lml_fresh = model_gp_fresh.log_marginal_likelihood_value_

            # Decide which one to keep: warm-start vs fresh
            if lml_fresh > lml:
                model_gp = model_gp_fresh
                lml = lml_fresh

        # Update "last good" kernel and LML for next iteration ---
        kernel_prev = model_gp.kernel_
        lml_prev = lml

        # Pf estimation with MCs
        x_mcs_pf = np.random.normal(0, 1, size=(int(n_mcs_pf), lstate.input_dim))
        mean_pf, _ = parallel_predict(model_gp, x_mcs_pf)
        Pf_model = float(np.mean(mean_pf < 0.0))
        Pf_rel_diff = (Pf_model - Pf_ref) / Pf_ref
        pf_evol.append(Pf_model)
        lml_evol.append(lml)

        # reliability index, B
        B_model = - norm.ppf(Pf_model)
        B_rel_diff = (B_model-B_ref)/B_ref

        print(f'Pf_model: {Pf_model:.3E}, Pf_rel_diff: {Pf_rel_diff:.2E}, B_rel_diff: {B_rel_diff:.2E}, LML = {lml:.2E}')
        wandb.log({"Pf_model":Pf_model, "Pf_rel_diff": Pf_rel_diff, "B_rel_diff": B_rel_diff, "LML": lml}, step=it)

        # Making predictions of mean and std for mc population 
        x_mc_pool = np.random.normal(0, 1, size=(int(n_mcs_pool), lstate.input_dim))
        mean_pred, std_pred = parallel_predict(model_gp, x_mc_pool)
        
        # arguments for sampling
        args_sampling = {'n_samples': 1, # Number of samples to select
                        'skip_indices': None} # Indices to skip in the pool
        
        # If the strategy is 'moo', we need to add the Pf estimate for reliability-based method
        if al_strategy == 'moo':
            if args_al['moo_method'] == 'reliability': 
                args_sampling['pf_estimate'] = Pf_model # Current Pf estimate for reliability method
        
        if al_strategy == 'reif2' or al_strategy == "portfolio":
            x_mc_pool_physical = isoprobabilistic_transform(x_mc_pool, lstate.standard_marginals, lstate.physical_marginals)
            args_sampling['input_candidates'] = x_mc_pool_physical

        if al_strategy == "eier":
            args_sampling['model_gp'] = model_gp
            args_sampling['candidate_pool'] = x_mc_pool
            args_sampling['z_seed'] = int(random_state.randint(0, 2**31 - 1))

        # Compute the indices to select based on the active learning strategy
        if args_al['pareto_metrics']:
            # If pareto metrics are enabled, we retrieve the pareto front and selected indices
            pareto, selected_indices, pmin, pmax = strategy.get_indices(
            mean_prediction=mean_pred,
            std_prediction=std_pred,
            **args_sampling
            )
            mean_pred_norm = normalize_array(np.abs(mean_pred))
            std_pred_norm = normalize_array(std_pred)
            selected_objective_norm = np.column_stack(
                (-mean_pred_norm[selected_indices], std_pred_norm[selected_indices])
            )
            # Saving points for pareto metrics (full Pareto front, and selected sample)
            denom = pmax - pmin
            denom = np.where(denom == 0.0, 1.0, denom)
            selected_local_norm = (selected_objective_norm - pmin) / denom
            pareto_metrics.append((pareto.tolist(), selected_local_norm.tolist()))
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
        selected_samples_norm = np.atleast_2d(np.asarray(selected_samples_norm, dtype=np.float64))
        selected_outputs = np.atleast_1d(np.asarray(selected_outputs, dtype=np.float64))
        x_train_norm = np.concatenate((x_train_norm, selected_samples_norm), axis=0)
        y_train = np.concatenate((y_train, selected_outputs), axis=0)

        # Saving results
        results_file['Pf_model'] = pf_evol

        if it % save_interval == 0:
            with open(results_dir + 'output.json', 'w') as file_id:
                        json.dump(results_file, file_id)

            # # Save the model (pickle)
            # with open(store_model_dir + 'gp_' + str(it) + '.pkl', 'wb') as file_id:
            #     pickle.dump(model_gp, file_id)

    # Saving final results
    results_file['Pf_model'] = pf_evol
    results_file['lml'] = lml_evol
    results_file['Pareto_metrics'] = pareto_metrics
    results_file['training_samples'] = x_train_norm.tolist(), y_train.tolist()  # training samples
    
    if al_strategy == "portfolio":
        results_file['portfolio_history'] = strategy.portfolio_history


    with open(results_dir + 'output.json', 'w') as file_id:
                    json.dump(results_file, file_id, indent=4)

    # # Save the model (pickle)
    # with open(store_model_dir + 'gp_' + "last" + '.pkl', 'wb') as file_id:
    #     pickle.dump(model_gp, file_id)

    end_time = time.time()
    execution_time = end_time - start_time
    wandb.finish()
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
