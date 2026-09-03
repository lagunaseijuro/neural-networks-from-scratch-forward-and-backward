"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x, dtype=float)

    indices = np.ndindex(*x.shape)

    for idx in indices:
        unit = np.zeros_like(x, dtype=float)
        unit[idx] = 1.0

        x_plus = x + eps * unit
        x_minus = x - eps * unit

        grad[idx] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    analytic_grad = np.asarray(analytic_grad, dtype=float)
    numeric_grad = np.asarray(numeric_grad, dtype=float)

    diff = np.abs(analytic_grad - numeric_grad)

    denom = np.maximum(
        np.abs(analytic_grad),
        np.abs(numeric_grad)
    )

    denom = np.maximum(
        denom,
        tol * np.ones_like(denom)
    )
    
    return float(np.max(diff / denom))

# Step 3 - make_dense (not yet solved)
# TODO: implement

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

