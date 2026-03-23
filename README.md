## word2vec (NumPy implementation)

Simple word2vec model implemented using numpy processing tools.
Covers the data cleaning process, reading plain-text from a .txt
file as well as a simple training process.

### Requirements

- Python 3.10+ (tested version)
- NumPy (see requirements.txt)

Install dependencies:

```bash
pip install -r requirements.txt
```

### Project structure

- `src/data_cleaning.py` – text loading, basic tokenisation, vocabulary
	and one-hot training data generation.
- `src/word2vec.py` – batched forward pass, loss, and gradient updates
	for the word2vec model.
- `src/train.py` – training loop with mini-batch SGD and loss tracking.
- `src/script.py` – entry point that wires the pieces together and
	prints simple training analytics.
- `data/` – example data files:
	- `stopwords.txt` – one stop word per line.

### Providing a dataset

1. Place your corpus file in the `data/` directory, e.g.
	 `data/my_corpus.txt`.
2. Optionally update `data/stopwords.txt` with one word per line to be
	 filtered out during preprocessing.
3. In `src/script.py`, change the FILE_NAME parameter

```python
FILE_NAME='data/your_file.txt'
```

### Running training

From the project root:

```bash
python -m src.script
```

Hyperparameters such as `WINDOW_SIZE`, `LEARNING_RATE`, `EPOCHS`,
`DIM`, and `BATCH_SIZE` can be adjusted at the top of `src/script.py`.

At the end of training, the script prints basic analytics including:

- number of training samples
- vocabulary size
- final and best loss values
- statistics of embedding vector norms

