import numpy as np
import joblib
import tokenizers

from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_tokenizer(path='./tokenizer/tokenizer.joblib'):
    return joblib.load(path)


def generate_sentence(model, tokenizer: tokenizers.Tokenizer, current_word, vocab_size, max_len=256):
    token_list = tokenizer.encode(f'[CLS]{current_word}').ids
    
    while True:
        encoded = pad_sequences([token_list], maxlen=max_len, padding='pre')
        
        result = np.argmax(model.predict(encoded, verbose=0))
        token_list.append(result)
        if len(token_list) > max_len:
            break
        
    
    sentence = tokenizer.decode(token_list)
    return sentence
