# TODO: Voting and accuracy functions on this file
import numpy as np


# Soft Voting function from np arrays of probabilities
def soft_voting(probs):
    return sum(probs) / len(probs)


# Hard Voting function from classifers decisions (0 or 1)
def hard_voting(decisions):
    return sum(decisions) / len(decisions) > 0.5


# Softmax function from np array of probabilities
def softmax(probs):
    exp_probs = np.exp(probs)
    return exp_probs / sum(exp_probs)  # Normalize to sum 1
