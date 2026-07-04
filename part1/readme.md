# Galton Heights: Expectation-Maximization (EM) Algorithm


This project implements the **Expectation-Maximization (EM) algorithm** from scratch in Python to train a **Gaussian Mixture Model (GMM)** on the Galton Heights dataset. The model learns to separate unlabeled height data into two distinct Gaussian distributions representing **mothers** and **adult sons**.

## Dataset

* **Dataset:** Galton Families Dataset (1886)
* **Features Used:** `mother` and `childHeight` (filtered for male children to ensure two distinct Gaussian distributions)
* The labels are ignored during training so the EM algorithm can automatically discover the two groups using soft assignment.

## Technologies

* Python
* NumPy (mathematical computations)
* Pandas (data loading)

## Project Structure

```text
.
└── part1/
    ├── GaltonFamilies.csv
    ├── main.py            # Entry point and live demo
    ├── em.py              # EM algorithm implementation
    ├── gaussian.py        # Gaussian PDF and log-likelihood functions
    ├── utils.py           # Data loading and preprocessing
    ├── requirements.txt
    └── README.md
```

## How to Run

1. Navigate to the project directory:

```bash
cd part1
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the program:

```bash
py main.py
```

## What the Program Does

* Loads and preprocesses the Galton Heights dataset.
* Initializes the Gaussian parameters.
* Runs the EM algorithm from scratch.
* Prints the optimization tracking table for Iterations 0, 1, and 2.
* Prompts the user to enter a test height for a live demonstration.
* Outputs the Bayesian posterior probabilities for the given height.

## Example Output

```text
Training EM Algorithm...

Iteration | Mu1 (Child) | Mu2 (Parent) | Var1^2 | Var2^2 | Pi1 | Pi2 | Log-Likelihood
0          | 59.00       | 78.00        | 11.73 | 11.73 | 0.50 | 0.50 | -6291.03
1          | 64.43       | 70.52        | 5.57  | 3.64  | 0.77 | 0.23 | -3727.34
2          | 64.48       | 70.46        | 5.89  | 3.98  | 0.77 | 0.23 | -3724.20

==================================================
MODEL READY FOR LIVE DEMO.
Type the height the coach gives you.
Type 'quit' to stop.
==================================================

Enter test height: 68

--- LIVE DEMO RESULTS FOR HEIGHT: 68.0 inches ---
Probability it is a Child  : 72.54%
Probability it is a Parent : 27.46%
------------------------------------------------
```

## Learning Outcomes

This project demonstrates:

* Implementing the Expectation-Maximization (EM) algorithm from scratch.
* Modeling data using Gaussian Mixture Models (GMMs).
* Computing posterior probabilities using Bayesian inference.
* Understanding the E-Step and M-Step of the EM algorithm.
* Tracking convergence through log-likelihood optimization.
* Demonstrating why probabilistic soft assignments are more effective than splitting data at the global mean.
