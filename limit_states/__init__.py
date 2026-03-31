from .g2d_four_branch_6 import g2D_four_branch_6
from .g2d_four_branch_7 import g2D_four_branch_7
from .g2d_hmcmc_himmelblau import g2d_himmelblau
from .g2d_hat import g2D_hat_function
from .g6d_nonlinear_oscillator import g6d_nonlinear_oscillator
from .g_high_dimensional import gd_high_dimensional
from .g8d_two_dof_oscillator import g8d_two_dof_oscillator
from .g1_wei import g2d_wei_g1
from .g2_wei import g2d_wei_g2
from .g3_wei import g3d_wei_g3

REGISTRY = {}

REGISTRY["four_branch_6"] = g2D_four_branch_6
REGISTRY["four_branch_7"] = g2D_four_branch_7
REGISTRY["himmelblau"] = g2d_himmelblau
REGISTRY["hat"] = g2D_hat_function
REGISTRY["nonlinear_oscillator"] = g6d_nonlinear_oscillator
REGISTRY["high_dimensional"] = gd_high_dimensional
REGISTRY["2dof_oscillator"] = g8d_two_dof_oscillator
REGISTRY["wei_g1"] = g2d_wei_g1
REGISTRY["wei_g2"] = g2d_wei_g2
REGISTRY["wei_g3"] = g3d_wei_g3
