import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from crawler import get_poem
from modules import build_model, sentence_generation


if not os.path.isdir('./poems'):
    os.mkdir('./poems')
if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')

# get poem with crawler
if len(os.listdir('./poems')) < 50:
    get_poem()

# read text
text = []
for i in range(1, 51):
    n = '1'
    if i < 10:
        n = '0' + str(i)
    else:
        n = str(i)

    with open('./poems/poem' + n + '.txt', 'r', encoding='utf-8') as f:
        while 1:
            tmp = f.readline()
            text.append(tmp[:-1])
            if tmp == '':
                break

# Tokenizer
t = Tokenizer()
t.fit_on_texts(text)
vocab_size = len(t.word_index) + 1

# make sequences
sequences = list()
for line in text:
    encoded = t.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i + 1]
        sequences.append(sequence)

idx2word = {}
for key, value in t.word_index.items():
    idx2word[value] = key

max_len = max(len(ln) for ln in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

# split sequences into X, Y
sequences = np.array(sequences)
X = sequences[:, :-1]
Y = sequences[:, -1]
Y = to_categorical(Y, num_classes=vocab_size)

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'cpkt')
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

model = build_model(vocab_size, max_len)

# train model
if len(os.listdir('./checkpoints')) == 0:
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, batch_size=256, epochs=250, callbacks=[checkpoint_callback], verbose=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))


# get word from input and generate text(50 words)
# input 'x' to stop
while True:
    word = input('시작 단어 입력(종료: x): ')
    if word == 'x':
        break
    else:
        text_len = int(input('단어 수 입력: '))
        print(sentence_generation(model, t, word, text_len))
        print('')
