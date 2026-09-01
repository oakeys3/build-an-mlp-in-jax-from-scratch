"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    key = jax.random.PRNGKey(seed)
    return key

# Step 2 - split_prng_key
import jax

def split_prng_key(key, num):
    return jax.random.split(key, num)

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    return jax.random.normal(key, shape)

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    shape = (batch_size, num_features)
    return sample_normal_matrix(key, shape)

# Step 5 - assign_class_labels
def assign_class_labels(inputs, num_classes):
    res = []
    batch_size, num_features = inputs.shape

    for b in range(batch_size):
        res.append(jnp.argmax(inputs[b][:num_classes]))
    
    return jnp.int32(res)

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(labels, num_classes):
    one_hot = (labels[:, None] == jnp.arange(num_classes)[None, :]).astype(jnp.float32)
    return one_hot

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    shape = (in_dim, out_dim)
    W = sample_normal_matrix(key, shape) * scale
    b = jnp.zeros(out_dim,)

    return {
        'W': W,
        'b': b
    }

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    subkeys = split_prng_key(key, len(layer_sizes))
    mlp_params = []

    for i in range(len(layer_sizes)-1):
        layer_key = subkeys[i]
        layer_in = layer_sizes[i]
        layer_out = layer_sizes[i+1]

        layer = init_linear_layer(subkeys[i], layer_in, layer_out, scale)

        mlp_params.append(layer)

    return mlp_params

# Step 9 - linear_forward
def linear_forward(x, layer_params):
    W = layer_params['W']
    b = layer_params['b']

    y = jnp.matmul(x, W) + b

    return y

# Step 10 - relu_activation
import jax.numpy as jnp


def relu_activation(x):
    x_relu = []
    for val in x:
        x_relu.append(jnp.maximum(val, 0.0))

    return jnp.array(x_relu)

# Step 11 - softmax_probabilities
import jax.numpy as jnp

def softmax_probabilities(logits):
    shifted_logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    exps = jnp.exp(shifted_logits)
    exp_sum = jnp.sum(exps, axis=-1, keepdims=True)

    softmax = exps / exp_sum
    
    return softmax

# Step 12 - mlp_forward (not yet solved)
# TODO: implement

# Step 13 - log_softmax_logits (not yet solved)
# TODO: implement

# Step 14 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 15 - classification_accuracy (not yet solved)
# TODO: implement

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

