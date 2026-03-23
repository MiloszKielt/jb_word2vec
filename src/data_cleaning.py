import re

with open('data/stopwords.txt', 'r') as f:
    stop_words = f.read().splitlines()

STOPWORDS = stop_words

def get_file_data(filename:str):
    
    file_contents = []
    with open(filename, 'r', encoding='utf-8') as f:
        file_contents = f.read()
        
    return file_contents
import numpy as np

def split_text_into_sentences(file_contents:list[str], stop_word_removal=False) -> list[str]:
    text = []
    for sentence in file_contents.split('.'):
        sentence_words = re.findall(r'\b\w+\b', sentence)
        
        line = ''
        
        for word in sentence_words:
            if stop_word_removal:
                if len(word) > 1 and word not in STOPWORDS:
                    line += ' ' + word
            else:
                if len(word) > 1:
                    line += ' ' + word
                    
        text.append(line)
        
    return text

def generate_dict_data(text:list[str]) -> list[dict | list[str] | int]:
    word_to_index = dict()
    index_to_word = dict()
    corpus = []
    count = 0
    vocab_size = 0
    
    for sentence in text:
        for word in sentence.split():
            word = word.lower()
            corpus.append(word)
            if word_to_index.get(word) == None:
                word_to_index.update( {word: count} )
                index_to_word.update( {count: word} )
                count += 1
    vocab_size = len(word_to_index)
    corpus_len = len(corpus)
    
    return {
        'word_to_index': word_to_index,
        'index_to_word': index_to_word,
        'corpus': corpus,
        'vocab_size': vocab_size,
        'corpus_len': corpus_len
    }


def get_one_hot_vectors(target:str, context:list[str], vocab_size:int, word_to_index:dict):
    
    target_vector = np.zeros(vocab_size)
    word_dict_idx = word_to_index.get(target)
    target_vector[word_dict_idx] = 1
    
    context_vector = np.zeros(vocab_size)
    
    for word in context:
        word_dict_idx = word_to_index.get(word)
        context_vector[word_dict_idx] = 1
        
    return target_vector, context_vector


def generate_training_data(corpus:list[str],window_size:int,vocab_size:int,word_to_index:dict[str:int],length_of_corpus:int,sample=False):
    training_data = []
    training_samples = []
    
    for i, word in enumerate(corpus):
        target_idx = i
        target = word
        context = []
        
        if i == 0:
            context = [corpus[x] for x in range(i + 1 , window_size + 1)] 
            
        elif i == length_of_corpus-1:
            context = [corpus[x] for x in range(length_of_corpus - 2, length_of_corpus - 2 - window_size, -1)]
            
        else:
            for x in range(target_idx-1, target_idx-1 - window_size, -1):
                if x >= 0:
                    context.extend([corpus[x]])
            
            for x in range(target_idx+1, target_idx+window_size):
                if x < length_of_corpus:
                    context.extend([corpus[x]])
                    
        target_vector, context_vector = get_one_hot_vectors(
            target=target,
            context=context,
            vocab_size=vocab_size,
            word_to_index=word_to_index
            )
        training_data.append([target_vector, context_vector])
        
        if sample:
            training_samples.append([target, context])
    
    return training_data, training_samples


def get_data(filename:str, window_size:int, stop_word_removal=False, sample=False):
    file_data = get_file_data(filename)
    text = split_text_into_sentences(file_data, stop_word_removal)
    vocabulary = generate_dict_data(text)
    training_vec, training_sample = generate_training_data(
        corpus=vocabulary['corpus'],
        window_size=window_size,
        vocab_size=vocabulary['vocab_size'],
        word_to_index=vocabulary['word_to_index'],
        length_of_corpus=vocabulary['corpus_len'],
        sample=sample
    )
    
    if sample:
        return training_vec, training_sample
    return training_vec
    