import numpy as np
def prox_l1(u, threshold):
    #L1 范数的近端算子 (Soft Thresholding)
    return np.sign(u) * np.maximum(np.abs(u) - threshold, 0)