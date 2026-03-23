
import numpy as np


def softmax(x):
    """Numerically stable softmax for batched inputs.

    Expects a 2D array of shape (batch_size, vocab_size).
    """

    x = np.asarray(x)
    x_max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=1, keepdims=True)


def forward(weight_input, weight_output, target_batch):
    """Batched forward pass.

    Parameters
    ----------
    weight_input : (vocab_size, embedding_dim)
    weight_output : (embedding_dim, vocab_size)
    target_batch : (batch_size, vocab_size)
        Batch of one-hot target word vectors.
    """

    x = np.asarray(target_batch)


    
    hidden_layer = x @ weight_input # (batch, embed_dim)
    u = hidden_layer @ weight_output # (batch, vocab)

    y_pred = softmax(u)
    return y_pred, hidden_layer, u


def backward(weight_input, weight_output, total_err, hidden_layer, target_batch, lr):
    """Batched backward pass and SGD update."""

    x = np.asarray(target_batch)
    e = np.asarray(total_err)
    h = np.asarray(hidden_layer)

    dl_weight_output = h.T @ e # (embed_dim, vocab)
    grad_hidden = e @ weight_output.T # (batch, embed_dim)
    dl_weight_input = x.T @ grad_hidden # (vocab, embed_dim)

    weight_input -= lr * dl_weight_input
    weight_output -= lr * dl_weight_output

    return weight_input, weight_output


def calc_error(y_pred, context):
    """Vectorised error computation for the multi-context loss.

    For each training example with ``k`` context words, the error for
    vocabulary index ``i`` is:

    - ``(k * p_i) - 1`` if ``i`` is a context word
    - ``k * p_i`` otherwise

    where ``p_i`` is the model probability for index ``i``.

    Both *y_pred* and *context* must be 2D arrays of shape
    (batch_size, vocab_size).
    """

    y = np.asarray(y_pred)
    c = np.asarray(context)
    
    k = c.sum(axis=1, keepdims=True)
    total_err = k * y.copy()
    total_err[c == 1] -= 1.0
    return total_err


def calc_loss(y_pred, context):
    """The loss calculation function

    With context mask ``c`` and probabilities ``p``:

        L = -sum_{i | c_i=1} p_i + k * log(sum_j exp(p_j))

    where ``k`` is the number of context words.
    """

    y = np.asarray(y_pred)
    c = np.asarray(context)

    k = c.sum(axis=1)
    sum_1 = -(y * c).sum(axis=1)
    sum_2 = k * np.log(np.exp(y).sum(axis=1))
    total_loss = sum_1 + sum_2
    return total_loss.sum()