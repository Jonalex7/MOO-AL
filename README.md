
# Batch Active Learning for Structural Reliability

In this repository, you can:

- Train a GP-based surrogate model through batch active learning. 

You can customize the active learning process, including target function, acquisition function, size of batch candidate samples, and experiment settings.

## To-do list

- [ ] Config files
- [ ] Parallel pf computation
- [ ] Separate active learning modules
- [x] Implemented MO selection

## Installation

TBD

## Usage

To run experiments, run:

```python
python main_batchactive.py [-h] [--ls [LS]] [--al_f [AL_F]] [--al_b [AL_B]] [--seed [SEED]] [--n_exp [N_EXP]]
```
## Arguments
* `-h, --help` 

  Shows a message listing the available arguments.

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
