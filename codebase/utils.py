import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def mse_measure(input, target):
    input = input.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    mses = []
    for row in range(input.shape[0]):
        mses.append(mean_squared_error(input[row], target[row]))
    return np.mean(mses)

def r2_measure(input, target):
    input = input.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    r2s = []
    for row in range(input.shape[0]):
        r2s.append(r2_score(input[row], target[row]))
    return np.mean(r2s)
