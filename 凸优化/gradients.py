import numpy as np
def grad_least_squares(A, b, x):
    #LASSO 问题中的最小二乘梯度
    return A.T @ (A @ x - b)
def grad_logistic_loss(X, y, w):
    #逻辑回归梯度
    m = X.shape[0]
    z = y * (X @ w)
    coef = -y / (1 + np.exp(z))
    return (X.T @ coef) / m