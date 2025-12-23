import time
from lasso_solver import solve_lasso
from logistic_solver import solve_logistic
def run_experiment(name, func):
    print(f"\n>>> Running {name} experiment...")
    start = time.perf_counter()
    x, hist = func()
    duration = time.perf_counter() - start
    if hist:
        print(f"[{name}] Results:")
        print(f" - Iterations: {len(hist['residual'])}")
        print(f" - Final Residual: {hist['residual'][-1]:.2e}")
        print(f" - CPU Time: {duration:.4f}s")
        print(f" - Plot saved as {name.lower()}_convergence.png")
if __name__ == '__main__':
    run_experiment("LASSO", solve_lasso)
    run_experiment("Logistic", solve_logistic)