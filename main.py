import datetime
import os
import argparse
import json
import pickle
import time

import numpy as np
import yaml
from scipy.stats import norm
import wandb

from limit_states import REGISTRY as ls_REGISTRY
from active_learning.active_learning import AcquisitionStrategy
from active_learning.eier import build_gp_cache_from_gpr, estimate_pf_posterior_samples
from utils.data import isoprobabilistic_transform, normalize_array, parallel_predict, resolve_cpu_workers
from utils.gp_training import (
    fit_gp_with_optional_stabilization,
    is_bad_fit,
    make_base_kernel,
    resolve_gp_alpha,
)


def _fmt_sci(value):
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.3E}"


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
    save_model_gp = bool(config.get("save_model_gp", config.get("model_gp", False)))
    eff_constant = float(config.get("eff_constant", 2.0))
    config["save_model_gp"] = save_model_gp
    # Drop legacy key to keep saved config.json consistent.
    if "model_gp" in config:
        del config["model_gp"]
    config['name_exp'] = name_exp

    # Loading limit state and ref. Pf
    lstate = ls_REGISTRY[casestudy]()
    Pf_ref = lstate.target_pf
    B_ref = - norm.ppf(Pf_ref)

    # results directory
    # New layout: results/<case_study>/<method>/<run_folder>/
    date_time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if al_strategy == 'moo':
        method_name = f'{al_strategy}_{config["moo_method"]}'
    else:
        method_name = al_strategy
    run_folder = f'{al_batch}_{name_exp}_{date_time_stamp}'
    results_dir = f'results/{casestudy}/{method_name}/{run_folder}/'

    store_model_dir = os.path.join(results_dir, "model")

    dirs_to_create = [results_dir]
    if save_model_gp:
        dirs_to_create.append(store_model_dir)

    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Store the evolution of Pf, pareto metrics, and training samples
    results_file = {}
    pf_evol = []
    pf_post_mean_evol = []
    pf_post_cov_evol = []
    pf_post_ci95_evol = []
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
    n_g_pf = int(config.get('n_g_pf', 1000))
    n_mcs_eier_int = None
    if al_strategy == "eier":
        n_mcs_eier_int = int(config.get('n_mcs_eier_int', int(n_mcs_pool)))
    pf_post_pool_size = int(config.get('n_pf_post_pool', 10000))
    pf_post_batch_size = int(config.get('pf_post_batch_size', 500))
    predict_batch_size = int(config.get('predict_batch_size', 10000))
    raw_cpu_workers = config.get('cpu_workers', None)
    if raw_cpu_workers is not None:
        resolved_cpu_workers = resolve_cpu_workers(raw_cpu_workers)
        predict_n_jobs = resolved_cpu_workers
        if al_strategy == "eier":
            eier_num_workers = resolved_cpu_workers
        config['cpu_workers'] = int(resolved_cpu_workers)
    else:
        predict_n_jobs = resolve_cpu_workers(config.get('predict_n_jobs', -1))
        if al_strategy == "eier":
            eier_num_workers = resolve_cpu_workers(config.get('eier_num_workers', 1))
    if al_strategy == "eier":
        eier_num_workers = max(1, int(eier_num_workers))
    config['n_g_pf'] = n_g_pf
    config['n_pf_post_pool'] = pf_post_pool_size
    config['pf_post_batch_size'] = pf_post_batch_size
    config['predict_batch_size'] = predict_batch_size
    config['predict_n_jobs'] = predict_n_jobs
    if al_strategy in {"eff", "portfolio"}:
        config['eff_constant'] = eff_constant
    if al_strategy == "eier":
        config['n_mcs_eier_int'] = n_mcs_eier_int
        config['eier_num_workers'] = eier_num_workers

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
    if al_strategy in {"eff", "portfolio"}:
        args_al['eff_constant'] = eff_constant
    # If moo strategy, add moo_method
    if al_strategy == 'moo':
        args_al['moo_method'] = config['moo_method']    # 'knee', 'compromise', 'reliability', 'linear_decay'
        # If moo_reliability strategy, add relevant parameters
        if args_al['moo_method'] == 'reliability': 
            args_al['N_it'] = config['N_it']  # Number of iterations to consider for moving average
            args_al['delta_P0'] = config['delta_p0'] # (0,1) threshold of relative difference at which gamma=0.5
            args_al['k_balance'] = config['k_balance'] # Positive constant controlling how quickly gamma transition from 0 to 1
            args_al['input_dim'] = lstate.input_dim
    
        if args_al['moo_method'] == "linear_decay":
            args_al['eps_start'] = config['eps_start']  # Initial epsilon value
            args_al['eps_end'] = config['eps_end']      # Final epsilon value
            args_al['eps_T'] = config['eps_T']          # Number of iterations to decay over

    if al_strategy == "portfolio":
        args_al['portfolio_lambda'] = config['portfolio_lambda']  # Hedge balance (λ)
        args_al['portfolio_delta'] = config['portfolio_delta']    # Memory factor (δ)

    if al_strategy == "eier":
        args_al['batch_size_acq'] = config['batch_size_acq']
        args_al['n_z_mc'] = n_g_pf
        args_al['jitter_stddev'] = config['obs_stddev']
        args_al['local_mis_topk'] = config['local_mis_topk']
        args_al['debug_acq'] = config.get('debug_acq', False)
        args_al['eier_num_workers'] = eier_num_workers
        args_al['z_chunk_size'] = int(config.get('z_chunk_size', 64))
        args_al['n_mcs_eier_int'] = n_mcs_eier_int
        
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
    print("Experiment settings:")
    print(f"  config                : {config}")
    print(f"  reference_Pf          : {_fmt_sci(Pf_ref)}")
    print(f"  candidate_pool/iter   : {int(n_mcs_pool)}")
    print(f"  save_model_gp         : {save_model_gp}")
    if al_strategy == "eier":
        print(f"  eier_integration/iter : {int(n_mcs_eier_int)}")
    print(
        f"  posterior_pf_samples  : N_g={n_g_pf} | "
        f"support={pf_post_pool_size} | "
        f"batch={pf_post_batch_size}\n"
    )
    # Active learning loop
    for it in range(iterations + 1):
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
            gp_alpha = resolve_gp_alpha(config, y_train)

        model_gp, gp_fit_info = fit_gp_with_optional_stabilization(
            x_train_norm=x_train_norm,
            y_train=y_train,
            init_kernel=init_kernel,
            input_dim=lstate.input_dim,
            gp_alpha=gp_alpha,
            n_restarts_optimizer=0,  # refine around warm-start
            enable_stabilization=(al_strategy == "eier"),
        )
        lml = model_gp.log_marginal_likelihood_value_
        
        if is_bad_fit(lml, lml_prev, lml_drop_tol=50.0, abs_lml_low=-100.0):
            # This fit looks suspicious -> try a fresh base kernel with restarts
            base_kernel = make_base_kernel(lstate.input_dim)
            model_gp_fresh, gp_fit_info_fresh = fit_gp_with_optional_stabilization(
                x_train_norm=x_train_norm,
                y_train=y_train,
                init_kernel=base_kernel,
                input_dim=lstate.input_dim,
                gp_alpha=gp_alpha,
                n_restarts_optimizer=9,  # full search from scratch
                enable_stabilization=(al_strategy == "eier"),
            )
            lml_fresh = model_gp_fresh.log_marginal_likelihood_value_

            # Decide which one to keep: warm-start vs fresh
            if lml_fresh > lml:
                model_gp = model_gp_fresh
                lml = lml_fresh
                gp_fit_info = gp_fit_info_fresh

        # Update "last good" kernel and LML for next iteration ---
        kernel_prev = model_gp.kernel_
        lml_prev = lml

        pf_post_rng = np.random.RandomState(int(random_state.randint(0, 2**31 - 1)))
        gp_cache = build_gp_cache_from_gpr(model_gp)
        _, pf_post_mean, pf_post_cov, pf_post_ci95 = estimate_pf_posterior_samples(
            cache=gp_cache,
            N_g=n_g_pf,
            batch_size_acq=pf_post_batch_size,
            rng=pf_post_rng,
            n_pool_pf=pf_post_pool_size,
            input_dim=lstate.input_dim,
        )
        pf_post_mean_evol.append(pf_post_mean)
        pf_post_cov_evol.append(pf_post_cov)
        pf_post_ci95_evol.append([pf_post_ci95[0], pf_post_ci95[1]])

        # Pf estimation with MCs
        x_mcs_pf = np.random.normal(0, 1, size=(int(n_mcs_pf), lstate.input_dim))
        mean_pf, _ = parallel_predict(
            model_gp,
            x_mcs_pf,
            n_jobs=predict_n_jobs,
            batch_size=predict_batch_size,
        )
        Pf_model = float(np.mean(mean_pf < 0.0))
        Pf_rel_diff = (Pf_model - Pf_ref) / Pf_ref
        pf_evol.append(Pf_model)
        lml_evol.append(lml)

        # reliability index, B
        B_model = - norm.ppf(Pf_model)
        B_rel_diff = (B_model-B_ref)/B_ref

        print(f"Iteration {it:02d}")
        print(f"  train_size            : {len(x_train_norm)}")
        print(
            f"  Pf_mean_predictor     : Pf={_fmt_sci(Pf_model)} | "
            f"rel_diff={_fmt_sci(Pf_rel_diff)} | "
            f"B_rel_diff={_fmt_sci(B_rel_diff)}"
        )
        print(
            f"  Pf_posterior_samples  : mean={_fmt_sci(pf_post_mean)} | "
            f"CoV={_fmt_sci(pf_post_cov)} | "
            f"CI95=[{_fmt_sci(pf_post_ci95[0])}, {_fmt_sci(pf_post_ci95[1])}]"
        )
        if al_strategy == "eier":
            print(
                f"  gp_stabilized         : {gp_fit_info['stabilized']} | "
                f"alpha_used={_fmt_sci(gp_fit_info['alpha_used'])} | "
                f"retries={gp_fit_info['retry_count']} | "
                f"kernel_upper={_fmt_sci(gp_fit_info['kernel_upper'])}"
            )
        metrics_payload = {
            "Pf_model": Pf_model,
            "Pf_rel_diff": Pf_rel_diff,
            "B_rel_diff": B_rel_diff,
            "LML": lml,
            "Pf_post_mean": pf_post_mean,
            "Pf_post_CoV": pf_post_cov,
            "Pf_post_CI95_low": pf_post_ci95[0],
            "Pf_post_CI95_high": pf_post_ci95[1],
        }
        if al_strategy == "eier":
            metrics_payload.update(
                {
                    "gp_stabilized": int(bool(gp_fit_info["stabilized"])),
                    "gp_alpha_used": float(gp_fit_info["alpha_used"]),
                    "gp_retry_count": int(gp_fit_info["retry_count"]),
                    "gp_kernel_upper": float(gp_fit_info["kernel_upper"]),
                }
            )
        wandb.log(metrics_payload, step=it)

        # Stop acquisition once the requested training budget is reached.
        # We still keep the Pf estimate computed at this final train size.
        if len(x_train_norm) >= total_samples:
            print(
                f"Reached total_samples={int(total_samples)} with train_size={len(x_train_norm)}. "
                "Stopping acquisition after final Pf estimate."
            )
            print("")
            break

        # Making predictions of mean and std for mc population 
        x_mc_pool = np.random.normal(0, 1, size=(int(n_mcs_pool), lstate.input_dim))
        x_mc_pool = np.asarray(x_mc_pool, dtype=np.float64)
        mean_pred, std_pred = parallel_predict(
            model_gp,
            x_mc_pool,
            n_jobs=predict_n_jobs,
            batch_size=predict_batch_size,
        )
        
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
        print("")

        # Saving results
        results_file['Pf_model'] = pf_evol
        results_file['Pf_post_mean'] = pf_post_mean_evol
        results_file['Pf_post_CoV'] = pf_post_cov_evol
        results_file['Pf_post_CI95'] = pf_post_ci95_evol

        if it % save_interval == 0:
            with open(results_dir + 'output.json', 'w') as file_id:
                        json.dump(results_file, file_id)

            if save_model_gp:
                model_path = os.path.join(store_model_dir, f"gp_{it}.pkl")
                with open(model_path, 'wb') as file_id:
                    pickle.dump(model_gp, file_id)

    # Saving final results
    results_file['Pf_model'] = pf_evol
    results_file['Pf_post_mean'] = pf_post_mean_evol
    results_file['Pf_post_CoV'] = pf_post_cov_evol
    results_file['Pf_post_CI95'] = pf_post_ci95_evol
    results_file['lml'] = lml_evol
    results_file['Pareto_metrics'] = pareto_metrics
    results_file['training_samples'] = x_train_norm.tolist(), y_train.tolist()  # training samples
    
    if al_strategy == "portfolio":
        results_file['portfolio_history'] = strategy.portfolio_history


    with open(results_dir + 'output.json', 'w') as file_id:
                    json.dump(results_file, file_id, indent=4)

    if save_model_gp:
        model_path = os.path.join(store_model_dir, "gp_last.pkl")
        with open(model_path, 'wb') as file_id:
            pickle.dump(model_gp, file_id)

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
