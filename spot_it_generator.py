import numpy as np
from design_templates import m2


def symbols_per_card(design):
    return np.max(np.sum(design, axis=0))


print(symbols_per_card(m2))
