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
    std = np.sqrt(2 / in_dim)
    W = np.random.randn(in_dim, out_dim) * std
    b = np.zeros(out_dim)

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

# Step 9 - make_optimizer
def compute_accuracy(model, X, y):
  logits, _ = model['forward'](X)
  return float(np.mean(np.argmax(logits, axis=1) == y))


def make_optimizer(params, lr=1e-2, kind='sgd', weight_decay=0.0):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    def step(grads):
      def update(p, g):
        if isinstance(p, dict):
          for key in p.keys():
            update(p[key], g[key])
        elif isinstance(p, list):
          for idx in range(len(p)):
            update(p[idx], g[idx])
        else:
          p -= lr * (g + weight_decay * p)
      
      update(params, grads)
        
    return {'step' : step}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # TODO: your approach here
    logits, caches = model['forward'](x_batch)

    loss, d_logits = loss_fn(logits, y_batch)

    dx, grads = model['backward'](d_logits, caches)

    optimizer['step'](grads)

    return loss

# Step 11 - train
import numpy as np

def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0, x_val=None, y_val=None, patience=None):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    # TODO: your approach here
    np.random.seed(seed)
    history = []

    best_val_accuracy = -1.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
      indices = np.random.permutation(np.arange(len(x)))
      X_shuffled = x[indices]
      y_shuffled = y[indices]

      total_loss = 0

      for idx in range(0, len(X_shuffled), batch_size):
        X_batch = X_shuffled[idx:idx + batch_size]
        y_batch = y_shuffled[idx:idx + batch_size]

        total_loss += train_step(model, loss_fn, optimizer, X_batch, y_batch)

      history.append(total_loss)

      if x_val is not None and y_val is not None:
        val_accuracy = compute_accuracy(model, x_val, y_val)

        if val_accuracy > best_val_accuracy:
          best_val_accuracy = val_accuracy
          patience_counter += 1
        else:
          patience_counter += 1

          if patience is not None and patience_counter >= patience:
            break




      
      



    return history

# Step 12 - design_network
def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task."""
    np.random.seed(seed)
    
    def generate_circles(n_samples, input_dim, noise=0.15, seed=0):
      np.random.seed(seed)
      n_half = n_samples // 2
      remainder = n_samples % 2

      X = np.zeros((n_samples, input_dim))

      r1, r2 = 10, 100
      theta1 = np.random.uniform(0, 2 * np.pi, n_half)
      theta2 = np.random.uniform(0, 2 * np.pi, n_half + remainder)

      x1, y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
      x2, y2 = r2 * np.cos(theta2), r2 * np.sin(theta2)

      first_col, second_col = np.hstack((x1, x2)), np.hstack((y1, y2))

      X[:, 0] = first_col
      X[:, 1] = second_col

      y = np.hstack((np.zeros(n_half), np.ones(n_half + remainder))).astype(int)

      X = X + noise * np.random.randn(n_samples, input_dim)

      indices = np.random.permutation(np.arange(n_samples))
      X = X[indices]
      y = y[indices]

      return X, y
    
    N = 500
    X, y = generate_circles(N, input_dim, noise=0.15, seed=seed)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X, y)
    linear_acc = lr.score(X, y)
    assert linear_acc < 0.82, f"Linear accuracy {linear_acc} too high!"
    
    hidden_dim = 10
    model = make_sequential([
        make_dense(input_dim, hidden_dim, initialize_weights),
        make_activation('relu'),
        make_dense(hidden_dim, hidden_dim, initialize_weights),
        make_activation('relu'),
        make_dense(hidden_dim, num_classes, initialize_weights)
    ])
    
    optimizer = make_optimizer(model['params'], lr=0.01)
    loss_fn = make_loss('cross_entropy')
    
    history = train(model, loss_fn, optimizer, X, y, 
                    epochs=300, batch_size=32, seed=seed)
    
    logits, _ = model['forward'](X)
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == y)
    assert accuracy >= 0.90, f"Accuracy {accuracy} < 0.90!"
    
    return model, {
        'accuracy': float(accuracy),
        'x': X,
        'y': y
    }

# Step 13 - improve_generalization
def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    """Improve held-out accuracy over an unregularized baseline.

    Inputs:
      baseline_model_fn: zero-arg callable -> fresh untrained sequential model
        (dict with 'forward', 'backward', 'params') matching the data dims.
      x_train, y_train: training features (N, D) and int labels (N,).
      x_val, y_val: validation features (N_val, D) and int labels (N_val,).
      seed: int for deterministic training.

    Returns:
      dict with keys:
        'val_accuracy': float accuracy of the improved model on x_val/y_val
        'baseline_val_accuracy': float val accuracy of plain unregularized SGD
        'predictions': np.ndarray shape (N_val,) int preds from improved model
        'model': the trained improved model

    Required behavior:
      val_accuracy > baseline_val_accuracy
      predictions == argmax(model.forward(x_val), axis=1)
      val_accuracy == mean(predictions == y_val)
      predictions are non-constant (not a trivial single-class predictor)
    """
    # Baseline Model
    np.random.seed(seed)

    baseline_model = baseline_model_fn()

    loss_fn = make_loss()
    optimizer = make_optimizer(baseline_model['params'], lr=0.01)

    train(baseline_model, loss_fn, optimizer, x_train, y_train, epochs=100, batch_size=32, seed=seed)

    based_logits, _ = baseline_model['forward'](x_val)
    based_preds = np.argmax(based_logits, axis=1)
    correct = np.sum((based_preds == y_val))
    baseline_val_accuracy = correct / len(x_val)

    # Improved Model
    np.random.seed(seed + 1)

    improved_model = baseline_model_fn()

    loss_fn = make_loss()
    optimizer = make_optimizer(improved_model['params'], lr=0.01, weight_decay=0.001)

    train(improved_model, loss_fn, optimizer, x_train, y_train, epochs=200, batch_size=64, seed=seed)

    impr_logits, _ = improved_model['forward'](x_val)
    predictions = np.argmax(impr_logits, axis=1)
    correct = np.sum(predictions == y_val)
    val_accuracy = correct / len(x_val)

    return {
      'val_accuracy' : float(val_accuracy),
      'baseline_val_accuracy' : float(baseline_val_accuracy),
      'predictions' : predictions,
      'model' : improved_model
    }

