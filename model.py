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

# Step 3 - make_dense
import numpy as np

def make_dense(in_dim, out_dim, weight_init_fn):
    W, b = weight_init_fn(in_dim, out_dim)

    layer = {
      'params' : {
        'W' : W,
        'b' : b 
      },
      'forward' : lambda : None,
      'backward' : lambda : None
    }

    def forward(x, cache=None):
      W, b = layer['params']['W'], layer['params']['b']

      y = x @ W + b 
      cache = (x.copy(), W.copy(), b.copy())

      return y, cache


    def backward(dout, cache):
      x, W, b = cache

      dx = dout @ W.T
      dW = x.T @ dout
      db = dout.sum(axis=0)

      grads = {
        'W' : dW,
        'b' : db
      }

      return dx, grads

    layer['forward'] = forward
    layer['backward'] = backward

    return layer

# Step 4 - make_activation
import numpy as np

def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # TODO: your approach here
    layer = {
    'params': {},  
    'forward': None,
    'backward': None
    }


    def forward(x, cache=None):
      y = np.maximum(x, 0)

      cache = x

      return y, cache


    def backward(dout, cache):
      x = cache

      dx = dout * (x > 0)

      return dx, {}


    layer['forward'] = forward
    layer['backward'] = backward

    return layer

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    if scheme == 'he':
        std = np.sqrt(2 / in_dim)
        W = np.random.randn(in_dim, out_dim) * std
        b = np.zeros(out_dim)
    else:
        pass
    
    return W, b

# Step 6 - make_loss
import numpy as np

def make_loss(kind='cross_entropy'):
    def loss_fn(logits, labels):
        logits = np.asarray(logits)
        labels = np.asarray(labels)

        batch, C = logits.shape

        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        sum_exp = np.sum(np.exp(shifted), axis=1, keepdims=True)
        log_probs = shifted - np.log(sum_exp)   

        correct_log_probs = log_probs[np.arange(batch), labels]
        loss = -np.mean(correct_log_probs)
        probs = np.exp(log_probs)
        grad = probs.copy()
        grad[np.arange(batch), labels] -= 1
        grad /= batch

        return float(loss), grad

    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # TODO: your approach here
    def forward(x):
      caches = []
      current = x

      for layer in layers:
        y, cache = layer['forward'](current)

        current = y 
        caches.append(cache)

      return current, caches 

  
    def backward(dout, caches):
      dx = dout 
      grads_list = []

      for layer, cache in zip(reversed(layers), reversed(caches)):
        dx, grad = layer['backward'](dx, cache)
        grads_list.append(grad)

      grads_list = grads_list[::-1]

      return dx, grads_list

    return {
      'forward' : forward,
      'backward' : backward,
      'params' : [layer['params'] for layer in layers]
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    logits, caches = model['forward'](x)

    loss, d_logits = loss_fn(logits, y)

    dx, param_grads = model['backward'](d_logits, caches)

    return loss, param_grads

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

