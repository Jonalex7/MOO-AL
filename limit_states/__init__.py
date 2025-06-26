from .g2d_four_branch_6 import g2D_four_branch_6
from .g2d_four_branch_7 import g2D_four_branch_7
from .g2d_hmcmc_himmelblau import g2d_himmelblau
from .g8d_dampedoscillator import g8d_damped_oscillator
from .g2d_hat import g2D_hat_function
from .g2d_rp55 import g2D_rp55
from .g21d_le_frame import g21D_rp201
from .g6d_nonlinear_oscillator import g6d_nonlinear_oscillator
from .g_high_dimensional import gd_high_dimensional
# from .g_fem_pushover import g_pushover

REGISTRY = {}

REGISTRY["four_branch_6"] = g2D_four_branch_6
REGISTRY["four_branch_7"] = g2D_four_branch_7
REGISTRY["himmelblau"] = g2d_himmelblau
REGISTRY["damped_oscillator"] = g8d_damped_oscillator
REGISTRY["hat"] = g2D_hat_function
REGISTRY["rp55"] = g2D_rp55
REGISTRY["le_frame"] = g21D_rp201
REGISTRY["nonlinear_oscillator"] = g6d_nonlinear_oscillator
REGISTRY["high_dimensional"] = gd_high_dimensional
# REGISTRY["pushover_frame"] = g_pushover