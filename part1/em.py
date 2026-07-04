import numpy as np
from gaussian import gaussian_pdf, calculate_log_likelihood

def run_em(X, max_iter=100, print_table=True):
    """Executes the Expectation-Maximization algorithm from scratch."""
    np.random.seed(42) # Seed fixed so your presentation table matches perfectly
    
    # INITIALIZATION
    mu1 = np.min(X) + 1.0  
    mu2 = np.max(X) - 1.0  
    var1 = np.var(X)       
    var2 = np.var(X)
    pi1 = 0.5             
    pi2 = 0.5
    
    tracking_metrics = []
    prev_ll = -np.inf # Keep track of the PREVIOUS log-likelihood
    
    for i in range(max_iter):
        # --- EXPECTATION STEP ---
        numerator1 = pi1 * gaussian_pdf(X, mu1, var1)
        numerator2 = pi2 * gaussian_pdf(X, mu2, var2)
        denominator = numerator1 + numerator2 + 1e-300
        
        gamma1 = numerator1 / denominator  
        gamma2 = numerator2 / denominator  
        
        # --- MAXIMIZATION STEP ---
        N1 = np.sum(gamma1)
        N2 = np.sum(gamma2)
        
        mu1_new = np.sum(gamma1 * X) / N1
        mu2_new = np.sum(gamma2 * X) / N2
        var1_new = np.sum(gamma1 * (X - mu1_new)**2) / N1
        var2_new = np.sum(gamma2 * (X - mu2_new)**2) / N2
        pi1_new = N1 / len(X)
        pi2_new = N2 / len(X)
        
        # Calculate Log-Likelihood for tracking
        ll = calculate_log_likelihood(X, mu1, mu2, var1, var2, pi1, pi2)
        
        # Save metrics for Iteration 0, 1, and 2 (Removed the [0] that caused the crash)
        if i <= 2:
            tracking_metrics.append([i, mu1, mu2, var1, var2, pi1, pi2, ll])
            
        # Update parameters for next iteration
        mu1, mu2, var1, var2, pi1, pi2 = mu1_new, mu2_new, var1_new, var2_new, pi1_new, pi2_new
        
        # Convergence check (compares current LL to PREVIOUS LL, not itself)
        if i > 0 and abs(ll - prev_ll) < 1e-6:
            break
            
        prev_ll = ll # Update previous LL for the next loop

    # Print the Tracking Table nicely
    if print_table:
        print("Iteration | Mu1 (Child) | Mu2 (Parent) | Var1^2 | Var2^2 | Pi1 | Pi2 | Log-Likelihood")
        for row in tracking_metrics:
            print(f"    {row[0]}     |   {row[1]:.2f}    |    {row[2]:.2f}     | {row[3]:.2f}  | {row[4]:.2f}  | {row[5]:.2f} | {row[6]:.2f} | {row[7]:.2f}")

    return mu1, mu2, var1, var2, pi1, pi2