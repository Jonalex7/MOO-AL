from .g2d_four_branch import g2D_four_branch
from .g2d_hmcmc_himmelblau import g2d_himmelblau
from .g8d_dampedoscillator import g8d_damped_oscillator
from .g2d_hat import g2D_hat_function
from .g2d_rp55 import g2D_rp55
from .g21d_le_frame import g21D_rp201
# from .g_fem_pushover import g_pushover

REGISTRY = {}

REGISTRY["four_branch"] = g2D_four_branch
REGISTRY["himmelblau"] = g2d_himmelblau
REGISTRY["damped_oscillator"] = g8d_damped_oscillator
REGISTRY["hat"] = g2D_hat_function
REGISTRY["rp55"] = g2D_rp55
REGISTRY["le_frame"] = g21D_rp201
# REGISTRY["pushover_frame"] = g_pushover