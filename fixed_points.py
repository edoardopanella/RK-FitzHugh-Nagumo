import numpy as np

# Newton's method
def newton(f, f_prime, x0, tol=1e-9, max_iter=100):

    iterations = [x0]  # include initial guess
    
    for i in range(max_iter):
        fpx = f_prime(x0)
        if fpx == 0:
            raise ZeroDivisionError("Derivative is zero; Newton step undefined.")
        
        x1 = x0 - f(x0)/fpx
        iterations.append(np.float64(x1))
        
        if abs(x1 - x0) < tol:
            return x1, iterations
        
        x0 = x1

    x0 = np.float64(x0)

    iterations = np.array(iterations)
    
    return x0, iterations

# Newton solve wrapper

def fixed_points(a, b, I, V0=0):
    """
    Computes (V*, W*) for given input current I
    using Newton's method on the cubic equation.
    """

    # Define the cubic polynomial
    def f(v):
        return v - (v**3)/3 - (1/b)*(v+a) + I
    
    def f_prime(v):
        return 1 - v**2 - (1/b)

    # Solve for V*
    v_star, iters = newton(f, f_prime, V0)

    # Compute W*
    w_star = (v_star + a)/b

    return v_star, w_star, iters
