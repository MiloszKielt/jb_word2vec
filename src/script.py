import numpy as np

from data_cleaning import get_file_data, generate_dict_data, generate_training_data
from train import train

WINDOW_SIZE = 3
LEARNING_RATE = 0.02
EPOCHS = 1000
DIM = 15
BATCH_SIZE = 128
FILE_NAME = 'data/silimarillion_test.txt'

if __name__ == '__main__':
    file_data = get_file_data(FILE_NAME)
    text = [file_data]  # maintain original generate_dict_data usage
    vocabulary = generate_dict_data(text)
    word_to_index = vocabulary['word_to_index']
    index_to_word = vocabulary['index_to_word']
    corpus = vocabulary['corpus']
    vocab_size = vocabulary['vocab_size']
    length_of_corpus = vocabulary['corpus_len']

    training_vec, training_sample = generate_training_data(
        corpus, WINDOW_SIZE, vocab_size, word_to_index, length_of_corpus, True
    )

    weights_inp, weights_out, epoch_loss, weights_inp_history, weights_out_history = train(
        DIM,
        WINDOW_SIZE,
        LEARNING_RATE,
        EPOCHS,
        vocab_size,
        training_vec,
        disp=True,
        interval=100,
        batch_size=BATCH_SIZE,
    )
    
    print("\n=== Training analytics ===")
    print(f"Training samples: {len(training_vec)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Epochs: {EPOCHS}")

    final_loss = epoch_loss[-1]
    best_epoch = int(np.argmin(epoch_loss))
    best_loss = epoch_loss[best_epoch]
    print(f"Final loss: {final_loss:.4f}")
    print(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")

    embed_norms = np.linalg.norm(weights_inp, axis=1)
    print(f"Embedding L2 norms -> mean: {embed_norms.mean():.4f}, std: {embed_norms.std():.4f}, "
          f"min: {embed_norms.min():.4f}, max: {embed_norms.max():.4f}")