from utils import load_and_prepare_data
from em import run_em
from gaussian import gaussian_pdf

def predict_posterior(test_height, mu1, mu2, var1, var2, pi1, pi2):
    """Calculates exact posterior probabilities for a coach's test height."""
    x = [[test_height]]
    
    p1 = pi1 * gaussian_pdf(x, mu1, var1)
    p2 = pi2 * gaussian_pdf(x, mu2, var2)
    
    total = p1 + p2
    
    # .item() safely converts any numpy array to a standard Python float
    prob_child = (p1 / total).item()
    prob_parent = (p2 / total).item()
    
    print(f"\n--- LIVE DEMO RESULTS FOR HEIGHT: {test_height} inches ---")
    print(f"Probability it is a Child : {prob_child * 100:.2f}%")
    print(f"Probability it is a Parent: {prob_parent * 100:.2f}%")
    print("--------------------------------------------------------")
    
    return prob_child, prob_parent

if __name__ == "__main__":
    # 1. Load Data
    filepath = "GaltonFamilies.csv"
    X = load_and_prepare_data(filepath)
    
    # 2. Run EM and print the required tracking table
    print("Training EM Algorithm...\n")
    mu1, mu2, var1, var2, pi1, pi2 = run_em(X)
    
    # 3. Live Demo Loop
    print("\n" + "="*50)
    print("MODEL READY FOR LIVE DEMO.")
    print("Type the height the coach gives you.")
    print("Type 'quit' to stop.")
    print("="*50)

    while True:
        user_input = input("\nEnter test height: ") 
        
        if user_input.lower() == 'quit':
            break
            
        try:
            height = float(user_input)
            
            # --- ROBUST VALIDATION ---
            # Data-driven bounds: Dataset min is 57", max is 79". 
            # We add a 5-inch buffer for edge cases.
            if height < 52 or height > 84:
                print("Invalid input. Height must fall within a plausible range for this dataset (52 to 84 inches).")
                continue  # Skips the prediction and asks again
                
            # ---------------------------
            
            predict_posterior(height, mu1, mu2, var1, var2, pi1, pi2)
            
        except ValueError:
            # Catches text, letters, or empty inputs
            print("Invalid input. Please enter a numerical height (e.g., 67.5)")