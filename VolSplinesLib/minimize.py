import numpy as np

def minimize(funcGrad, x0, bounds, maxiter=15000, ftol=1e-8, gtol=1e-5):
    """
    Perform bound-constrained minimization using the L-BFGS-B algorithm.

    Parameters:
    - funcGrad: Function that computes the objective function and its gradient.
    - x0: Initial guess for the variables.
    - bounds: List of tuples specifying the lower and upper bounds for each variable.
    - maxiter: Maximum number of iterations allowed.
    - ftol: Relative tolerance for the function value convergence criterion.
    - gtol: Tolerance for the gradient norm convergence criterion.

    Returns:
    - A dictionary containing the optimization result, including the solution, function value, and status.
    """
    n = len(x0)
    x = x0.copy()
    f, grad = funcGrad(x)
    
    m = 10
    iter = 0
    nfev = 1
    status = 0
    message = "Optimization terminated successfully."
    prev_f = f
    
    s_list = []
    y_list = []
    rho_list = []
    
    while iter < maxiter:
        q = grad.copy()
        k = len(s_list)
        alpha = np.zeros(k)
        
        for i in range(k - 1, -1, -1):
            alpha[i] = rho_list[i] * np.dot(s_list[i], q)
            q -= alpha[i] * y_list[i]
        
        r = q.copy()
        
        for i in range(k):
            beta = rho_list[i] * np.dot(y_list[i], r)
            r += s_list[i] * (alpha[i] - beta)
        
        p = -r
        
        for i in range(n):
            if bounds[i][0] == bounds[i][1]:
                p[i] = 0.0
            else:
                if x[i] <= bounds[i][0] and p[i] < 0:
                    p[i] = 0.0
                if x[i] >= bounds[i][1] and p[i] > 0:
                    p[i] = 0.0
        
        alpha_step = 1.0
        c1 = 1e-4
        c2 = 0.9
        max_linesearch = 20
        success = False
        
        for _ in range(max_linesearch):
            x_new = x + alpha_step * p
            
            x_new = np.maximum(x_new, [b[0] if b[0] > -np.inf else -np.inf for b in bounds])
            x_new = np.minimum(x_new, [b[1] if b[1] < np.inf else np.inf for b in bounds])
            
            f_new, grad_new = funcGrad(x_new)
            nfev += 1
            
            if f_new <= f + c1 * alpha_step * np.dot(grad, p):
                if np.dot(grad_new, p) >= c2 * np.dot(grad, p):
                    success = True
                    break
            
            alpha_step *= 0.5
        
        if not success:
            status = 1
            message = "Line search failed."
            break
        
        s = x_new - x
        y = grad_new - grad
        ys = np.dot(y, s)
        
        if ys > 1e-10:
            if len(s_list) == m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / ys)
        
        x = x_new.copy()
        f = f_new
        grad = grad_new.copy()
        
        if np.max(np.abs(grad)) < gtol:
            status = 0
            message = "Optimization terminated successfully (gtol)."
            break
        
        if abs(f - prev_f) < ftol * (1.0 + abs(f)):
            status = 0
            message = "Optimization terminated successfully (ftol)."
            break
        
        prev_f = f
        iter += 1
    
    if iter >= maxiter:
        status = 1
        message = "Maximum number of iterations exceeded."
    
    return {
        'x': x.copy(),
        'fun': f,
        'nfev': nfev,
        'nit': iter,
        'status': status,
        'message': message,
    }
