import numpy as np
def proximal_gradient_descent(grad_func, prox_func, x0, step_size, lambda_reg, max_iter=2000, tol=1e-9):
    #标准的近端梯度下降法 (ISTA)
    x = x0.copy()
    history = {'residual': []}
    for k in range(max_iter):
        # 1. 梯度下降步 (Gradient step)
        gradient_step = x - step_size * grad_func(x)
        # 2. 近端算子步 (Proximal step / Soft-thresholding)
        x_new = prox_func(gradient_step, step_size * lambda_reg)
        # 计算不动点残差 ||x_k+1 - x_k||
        residual = np.linalg.norm(x_new - x)
        history['residual'].append(residual)
        # 达到指定精度 10^-9 停止
        if residual < tol:
            break
        x = x_new.copy()
    return x, history