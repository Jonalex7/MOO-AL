import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
import yaml
from datetime import datetime
import pickle
import os
import wandb

from utils.data import Buildings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

config={}
#loading data set
case_study = config['case_study']  = 'u'

# ls_name = config['l_state'] = 'g2d_four_branch'
# dataset = torch.load('datasets/dataset_four_branch.pt')

ls_name = config['l_state'] = 'g2d_himmelblau'
dataset = torch.load('datasets/dataset_himmelblau_2e5.pt')

num_active_samples = [3, 5, 7, 10]

for active in num_active_samples:

    train_size = config['train_size'] =  10
    # seed = config['seed'] = None
    active_loop = config['active_loop'] = int(91/active) #to complete at least 100 samples
    active_samples = config['active_samples'] = active  #

    x_mc_test = dataset.input_tensor
    y_mc_test = dataset.output_tensor

    n_dataset_samples = len(y_mc_test)
    Pf_ref = torch.sum(y_mc_test < 0) / n_dataset_samples
    print(f' Target limit state: {ls_name}, Pf ref: {Pf_ref} ')

    results_dir = f'results/active_train/{ls_name}/{case_study}/{active_samples}/'
    date_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_norm = Buildings(x_mc_test, y_mc_test)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    seeds_exp= [391252418, 90523161, 375021598, 221860729, 45301975, 289396467, 698737664, 70359637, 800466323, 316421878]
    pool_size = len(dataset_norm.output_tensor) - train_size
    exp = 0

    for seed in seeds_exp:

        config['exp'] = exp

        if seed is not None:
            seed_experiment = seed
        else:
            seed_experiment = np.random.randint(0, 2**30 - 1)

        np.random.seed(seed_experiment)
        torch.manual_seed(seed_experiment)
        print(f'seed: {seed_experiment}')

        #Sample for initial training and validation
        train_ds, pool_ds = random_split(dataset_norm, [train_size, pool_size])
        idx_train = train_ds.indices
        idx_pool = pool_ds.indices
        idx_all = [i for i in range(len(dataset_norm.input_tensor))]

        wandb.init(project='2D_ActiveTrain',
            name=f'{ls_name}_'+case_study,
            config=config)
        
        results_file = {}
        pf_evol = []

        for ep in range(active_loop):

            x_train = dataset_norm.input_tensor[idx_train]
            y_train = dataset_norm.output_tensor[idx_train]
            x_mc_pool = dataset_norm.input_tensor[idx_pool]
            # ----
            kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
            gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
            gaussian_process.fit(x_train, y_train)
            gaussian_process.kernel_
            # Making predictions of mean and std for mc population 
            mean_prediction, std_prediction = gaussian_process.predict(dataset_norm.input_tensor, return_std=True)
            mean_pred = torch.tensor(mean_prediction)
            std_pred = torch.tensor(std_prediction)

            # # -----------------------------------------------
            Pf_model = torch.sum(mean_pred < 0) / len(mean_pred)
            print(f' Target limit state: {ls_name}, Pf ref: {Pf_ref:.3E}, Pf model: {Pf_model.item():.3E} ')
            print("Number of samples in the training set: ", len(x_train))
            print("Number of samples in the pool set: ", len(idx_pool))

            wandb.log({"pf_ref": Pf_ref, "pf_model": Pf_model}, step=ep)
            pf_evol.append(Pf_model)

            #geting samples with  U function
            u_function = (mean_pred.abs())/std_pred
            u_function[idx_train] = torch.inf
            _, u_idx = u_function.squeeze().topk(active_samples, largest=False)
            u_idx = u_idx.tolist()
            selected_idx_pool = [idx_all[idx] for idx in u_idx]
                
            print(f'selected idx: {selected_idx_pool}')

            idx_pool_ = [idx for idx in idx_pool if idx not in selected_idx_pool]
            idx_train_ = idx_train + selected_idx_pool

            pool_ds = Subset(dataset_norm, idx_pool_) 
            train_ds = Subset(dataset_norm, idx_train_)

            idx_pool = pool_ds.indices
            idx_train = train_ds.indices

            wandb.log({"train_size": len(train_ds)}, step=ep)

            results_file['seed'] = seed_experiment
            results_file['config'] = config
            results_file['model'] = gaussian_process
            results_file['idx_train'] = idx_train
            results_file['Pf_model'] = pf_evol

            with open(results_dir+'exp_'+str(exp)+'_'+date_time_stamp+'.pkl', 'wb') as file_id:
                pickle.dump(results_file, file_id)

            print(f'active loop {ep+1} \n')

        results_file['Pf_ref'] = Pf_ref
        results_file['model'] = gaussian_process
        with open(results_dir+'exp_'+str(exp)+'_'+date_time_stamp+'.pkl', 'wb') as file_id:
            pickle.dump(results_file, file_id)

        exp+=1

        # Close the wandb run 
        wandb.finish()

        print(f'Done! \n ')
    