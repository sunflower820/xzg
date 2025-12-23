import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_libsvm_data
from gradients import grad_least_squares
from proximal_operators import prox_l1
from proximal_gradient_solver import proximal_gradient_descent


def solve_lasso():
    A, b = load_libsvm_data('data/heart_scale.txt')
    if A is None: return None, None
    L = np.linalg.norm(A.T @ A, 2)
    step_size = 1.0 / L
    # 稍微调整 lambda_reg 获得更直的下降线
    x_opt, history = proximal_gradient_descent(
        lambda x: grad_least_squares(A, b, x),
        prox_l1, np.zeros((A.shape[1], 1)),
        step_size, 0.05, tol=1e-10
    )
    plt.figure(figsize=(8, 5))
    plt.semilogy(history['residual'][2:], color='#1f77b4', linewidth=2)
    plt.title("Convergence Curve: LASSO with L1 Regularization", fontsize=11)
    plt.xlabel("Iteration Count")
    plt.ylabel("Fixed Point Residual $||x_{k+1} - x_k||_2$")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig("lasso_convergence.png", dpi=300)
    plt.close()
    return x_opt, history