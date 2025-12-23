import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_libsvm_data
from gradients import grad_logistic_loss
from proximal_operators import prox_l1
from proximal_gradient_solver import proximal_gradient_descent


def solve_logistic():
    X, y = load_libsvm_data('data/heart_scale.txt')
    if X is None: return None, None
    m, n = X.shape
    L = np.linalg.norm(X.T @ X, 2) / (4 * m)
    step_size = 1.0 / L
    w_opt, history = proximal_gradient_descent(
        lambda w: grad_logistic_loss(X, y, w),
        prox_l1, np.zeros((n, 1)),
        step_size, 0.01, tol=1e-10
    )
    plt.figure(figsize=(8, 5))
    plt.semilogy(history['residual'][2:], color='#ff7f0e', linewidth=2)
    plt.title("Convergence Curve: Logistic Regression with L1", fontsize=11)
    plt.xlabel("Iteration Count")
    plt.ylabel("Fixed Point Residual")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig("logistic_convergence.png", dpi=300)
    plt.close()
    return w_opt, history