import numpy as np

def get_hidden_layer_activations(model, X, layer_index):
    """Calculate the activations of a specified hidden layer.
    
    Parameters:
    - model: Trained instance of sklearn.neural_network.MLPClassifier.
    - X: Input data, numpy array of shape (n_samples, n_features).
    - layer_index: Index of the hidden layer for which to compute activations.
    
    Returns:
    - Activations of the specified hidden layer, numpy array of shape (n_samples, n_units_in_layer).
    """
    if layer_index < 0 or layer_index >= len(model.coefs_) - 1:
        raise ValueError("Invalid layer_index.")
    
    # Forward propagate through the network until the specified layer
    activations = X
    for i in range(layer_index + 1):
        activations = np.dot(activations, model.coefs_[i]) + model.intercepts_[i]
        if i < len(model.coefs_) - 1: 
            activations = np.maximum(0, activations)
    
    return activations

mlp = # Fitted MLP Model
data = # Needed  data
i = # Index of hidden layer


# Retrieve activations for the ith hidden layer
embeddings = get_hidden_layer_activations(mlp, data, i)