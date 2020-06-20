import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


def build_model(vocab_size, max_len):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_len - 1))
    model.add(LSTM(256, return_sequences=True, kernel_initializer='glorot_uniform'))
    model.add(LSTM(256, return_sequences=True, kernel_initializer='glorot_uniform'))
    model.add(LSTM(256, kernel_initializer='glorot_uniform'))
    model.add(Dense(vocab_size, 'softmax', kernel_initializer='glorot_uniform'))
    return model


def sentence_generation(model, t, current_word, n):
    init_word = current_word
    sentence = ''
    for _ in range(n):
        encoded = t.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=73, padding='pre')
        result = np.argmax(model.predict(encoded, verbose=0))
        word = ''
        for word, index in t.word_index.items():
            if index == result:
                break
        current_word += ' ' + word
        sentence += ' ' + word
    sentence = init_word + sentence
    return sentence
