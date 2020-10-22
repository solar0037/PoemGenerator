import os
import numpy as np
import joblib
import tokenizers

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint

from crawl import get_poem
from utils import get_tokenizer, generate_sentence
from dataset_utils import get_dataset
from model import get_model


if __name__ == '__main__':
    # for tpu
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
    except:
        pass
    
    
    # make directories
    data_dir = './data'
    model_dir = './models'
    checkpoint_dir = './checkpoints'
    output_dir = './outputs'
    for dirname in [data_dir, model_dir, checkpoint_dir, output_dir]:
        if not os.path.isdir(dirname):
            os.mkdir(dirname)


    # get text
    if len(os.listdir(data_dir)) < 50:
        get_poem()
    
    # read text
    text = []
    for i in range(1, 50+1):
        # number of text -> filename
        n = f'0{str(i)}' if i < 10 else f'{str(i)}'  # 01, 02, 03, ..., 10, 11, 12, ...
        filename = f'{data_dir}/poem{n}.txt'

        with open(filename, 'r', encoding='utf-8') as f:
            while 1:
                line = f.readline()
                text.append(line[:-1])  # remove '\n'
                if line == '':
                    break

    # Tokenizer
    tokenizer: tokenizers.Tokenizer = get_tokenizer('./tokenizer/tokenizer.joblib')
    vocab_size = tokenizer.get_vocab_size()
    max_len = 256

    # make input sequences
    input_sequences = []
    for line in text:
        token_list = tokenizer.encode(f'[CLS]{line}[SEP]').ids
        # remove sentences longer than max_len
        if len(token_list) > max_len + 1:  # -1 because input max length is 512, and sentence will be split into 512, 1
            continue

        for i in range(2, len(token_list)):  # exclude ['[CLS]', 'word'] because prediction is imposiible
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    
    input_sequences = pad_sequences(input_sequences, maxlen=max_len + 1, padding='pre', truncating='pre')

    # make inputs, targets
    inputs = input_sequences[:, :-1]
    targets = input_sequences[:, -1].reshape(-1, 1)
    del input_sequences
    
    # train dataset
    train_dataset = get_dataset(inputs, targets)
    print(train_dataset)


    # model
    model: Model = get_model(vocab_size=vocab_size, max_len=max_len)
    model.summary()
    
    # optimizer, loss
    optimizer = Adam(learning_rate=1e-3)
    loss_object = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])

    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )

    model_filename = f'{model_dir}/model.h5'


    # train model
    if not os.path.exists(model_filename):
        try:
            model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        except:
            pass
        finally:
            model.fit(train_dataset, batch_size=None, epochs=250, callbacks=[checkpoint_callback])
            model.save(model_filename)
    
    # load model for inference
    else:
        model = tf.keras.models.load_model(model_filename)


    # generate text until [SEP] token
    while True:
        word = input('시작 단어 입력(종료: Ctrl + C): ')
        print(generate_sentence(model, tokenizer, word, vocab_size))
        print('')
