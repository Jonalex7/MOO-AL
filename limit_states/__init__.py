from .g2d_four_branch import g2D_four_branch
from .g2d_hmcmc_himmelblau import g2d_himmelblau
from .g_fem_pushover import g_pushover

REGISTRY = {}

REGISTRY["g2d_four_branch"] = g2D_four_branch
REGISTRY["g2d_himmelblau"] = g2d_himmelblau
REGISTRY["g4d_pushover_frame"] = g_pushover
