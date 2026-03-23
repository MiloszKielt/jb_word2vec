import numpy as np

from word2vec import forward, calc_error, backward, calc_loss


def train(
    word_embedding_dim,
    window_size,
    learning_rate,
    epochs,
    vocab_size,
    training_data,
    disp=False,
    interval=-1,
    batch_size=128,
):
    """Train word2vec model

    The one-hot training data list is stacked into NumPy arrays and
    processed in shuffled mini-batches. All math is fully vectorised and
    operates on 2D (batch, vocab_size) tensors.
    """

    weights_inp = np.random.uniform(-1, 1, (vocab_size, word_embedding_dim))
    weights_out = np.random.uniform(-1, 1, (word_embedding_dim, vocab_size))

    epoch_loss = []
    weights_inp_history = []
    weights_out_history = []

    # Stack into arrays once
    targets = np.stack([t for t, _ in training_data])
    contexts = np.stack([c for _, c in training_data])
    num_samples = targets.shape[0]

    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        total_loss = 0.0

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            batch_targets = targets[batch_idx]
            batch_contexts = contexts[batch_idx]

            y_pred, hidden_layer, _ = forward(weights_inp, weights_out, batch_targets)
            total_err = calc_error(y_pred, batch_contexts)
            weights_inp, weights_out = backward(
                weights_inp, weights_out, total_err, hidden_layer, batch_targets, learning_rate
            )

            batch_loss = calc_loss(y_pred, batch_contexts)
            total_loss += batch_loss

        epoch_loss.append(total_loss)

        if disp and interval != -1 and epoch % interval == 0:
            print(f'Epoch: {epoch}, Loss: {total_loss}')
            weights_inp_history.append(weights_inp.copy())
            weights_out_history.append(weights_out.copy())

    return weights_inp, weights_out, epoch_loss, weights_inp_history, weights_out_history