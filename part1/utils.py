import pandas as pd
import numpy as np

def load_and_prepare_data(filepath):
    """
    Loads the Galton dataset and combines Mothers and Sons 
    to create an unlabeled mixture of two Gaussian distributions.
    """
    df = pd.read_csv(filepath)
    
    # We compare Mothers and Adult Sons to ensure two distinct, unimodal Gaussian distributions
    mothers = df['mother'].dropna().values
    sons = df[df['gender'] == 'male']['childHeight'].dropna().values
    
    # Combine into one unlabeled dataset
    X = np.concatenate([mothers, sons])
    
    # Reshape to a 2D array for matrix math compatibility
    return X.reshape(-1, 1)