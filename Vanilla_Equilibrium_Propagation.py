import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# One-hot encoding
def create_target(t):
        target_vector = np.zeros(10)
        for i in range(10):
            if i == t:
                target_vector[i] = 1
        return target_vector

# Hard sigmoid [-1, 1]
def hsig(x):
    z = np.clip(x, -1, 1)
    return np.copy(z)

def d_hsig(x):
    z = (x > -1) & (x < 1)
    return np.copy(z)

# Load mini MNIST data
digits    = datasets.load_digits()
data      = digits.data
targets   = digits.target

# Standardize data
inputs = data - np.mean(data)
inputs = inputs/(np.std(data))

# Define network hyperparameters
n_x = 64
n_h = 50
n_y = 10

alpha1    = 0.01
alpha2    = 0.005
beta      = 1
epsilon   = 0.1

# Weight intialization
W1 = np.random.uniform(0, (4/(n_x + n_y)), (n_x, n_h))
W2 = np.random.uniform(0, (4/(n_h)), (n_h, n_y))

# Bias initialization
bh = np.random.uniform(0, 4/(n_x + n_y), n_h)
by = np.random.uniform(0, 4/(n_h), n_y)

for ex in range(5000):
    # Randomly sample from data
    rnd = np.random.randint(0, 1497)
    x = inputs[rnd]# + 0.1 * np.random.rand(64)
    t = create_target(targets[rnd])
    
    # Random activation initialization
    h = np.random.uniform(-1, 1, n_h)
    y = np.random.uniform(-1, 1, n_y)
    
    # Free Phase
    for itr in range(100):
        # Calculate free gradient steps
        dh = d_hsig(h) * (np.dot(x, W1) + np.dot(y, W2.T) + bh) - h
        dy = d_hsig(y) * (np.dot(h, W2) + by) - y
        
        # Update activations
        h = hsig(h + epsilon * dh)
        y = hsig(y + epsilon * dy)
        
    # Store free equilibrium states
    h_free = np.copy(h)
    y_free = np.copy(y)
    
    # Weakly Clamped Phase
    for itr in range(20):
        # Calculate weakly clamped gradient steps
        dy = d_hsig(y) * (np.dot(h, W2) + by) - y + beta * (t - y)
        dh = d_hsig(h) * (np.dot(x, W1) + np.dot(y, W2.T) + bh) - h
        
        # Update activations
        h = hsig(h + epsilon * dh)
        y = hsig(y + epsilon * dy)
        
    # Store weakly clamped activations
    h_clamped = np.copy(h)
    y_clamped = np.copy(y)
    
    # Update weights
    W1 += alpha1 * (1/beta) * (np.outer(x, h_clamped) - np.outer(x, h_free))
    W2 += alpha2 * (1/beta) * (np.outer(h_clamped, y_clamped) - np.outer(h_free, y_free))
    
    # Print Mean Squared Error
    if ex % 100 == 0:
        print(np.dot(t - y_free, t - y_free))
        
    # Learning rate schedule
    if ex % 2500 == 2499:
        alpha1 /= 10
        alpha2 /= 10

# Test Accuracy
score = 0
for test in range(200):
    rnd = np.random.randint(1497, 1797)
    x = inputs[rnd]
    t = create_target(targets[rnd])
    h = np.random.uniform(-1, 1, n_h)
    y = np.random.uniform(-1, 1, n_y)
    
    # Free Phase
    for itr in range(100):
        dh = d_hsig(h) * (np.dot(x, W1) + np.dot(y, W2.T) + bh) - h
        dy = d_hsig(y) * (np.dot(h, W2) + by) - y
        
        h = hsig(h + epsilon * dh)
        y = hsig(y + epsilon * dy)
        
    h_free = np.copy(h)
    y_free = np.copy(y)
    
    if np.argmax(y_free) == targets[rnd]:
        score += 1
    
print(score/200)

