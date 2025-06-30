from .g2d_four_branch_6 import g2D_four_branch_6
from .g2d_four_branch_7 import g2D_four_branch_7
from .g2d_hmcmc_himmelblau import g2d_himmelblau
from .g2d_hat import g2D_hat_function
from .g6d_nonlinear_oscillator import g6d_nonlinear_oscillator
from .g_high_dimensional import gd_high_dimensional

REGISTRY = {}

REGISTRY["four_branch_6"] = g2D_four_branch_6
REGISTRY["four_branch_7"] = g2D_four_branch_7
REGISTRY["himmelblau"] = g2d_himmelblau
REGISTRY["hat"] = g2D_hat_function
REGISTRY["nonlinear_oscillator"] = g6d_nonlinear_oscillator
REGISTRY["high_dimensional"] = gd_high_dimensional