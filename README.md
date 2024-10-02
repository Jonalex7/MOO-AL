
# Batch Active Learning with GP Regressor

Repository to run (`main_batchactive.py`) that trains a Gaussian Process Regressor using batch active learning. The script offers various options for customizing the active learning process, including the target function, acquisition function, size of batch candidate samples, and experiment settings.

## Usage

To run the script, use the following format:

```bash
python main_batchactive.py [-h] [--ls [LS]] [--al_f [AL_F]] [--al_b [AL_B]] [--seed [SEED]] [--n_exp [N_EXP]]
```
## Arguments
* `-h, --help` 

  Show the help message with the available arguments.

* ` --ls [LS]` 
  ### Target Limit State [LS]
  Specifies the target function to use in the active learning process. Options include:

  `four_branch` (default)
  
  `himmelblau` 
  
  `pushover_frame` (Opeensees 2D Pushover Analysis of 2-Story Moment Frame)

* `--al_f [AL_F]`
  ### Active Learning Strategy [AL_F]
  Specifies the strategy for batch active learning. Options include:

  `u_function` (default)
  
  `corr_det`
  
  `corr_eigen`

  `corr_entropy`

  `corr_condvar`
  
  `random`

* `--al_b [AL_B]`
  ### Size for a batch of candidate samples [AL_B]
  Defines the number of samples in each batch of candidates `int`. Default is 3.

* `--seed [SEED]`
  ### Random seed [SEED]
  Sets the random seed for reproducibility `int`. If not specified, a random seed will be used.

* `--n_exp [N_EXP]`
  ### Number of Experiments [N_EXP]
  Sets the number of experiments to run under the given settings `int`.  Default is 1.
