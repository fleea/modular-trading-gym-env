
from itertools import chain
import numpy as np

def get_multiplier(row: list):
    row = list(chain.from_iterable(row))
    rms_from_change = np.sqrt(np.mean(np.square(row)))
    return 1 / rms_from_change if rms_from_change != 0 else 1

def get_rms_multiplier(row: list):
    row = [x for x in row if not np.isnan(x)]
    rms_from_change = np.sqrt(np.mean(np.square(row)))
    return 1 / rms_from_change if rms_from_change != 0 else 1