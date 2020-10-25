import os
import joblib
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from model import get_model
from utils import generate_sentence


if __name__ == '__main__':
    checkpoint_dir = './checkpoints'
    model_dir = './models'

    vocab_size = 5000
    max_len = 256
    tokenizer = joblib.load('./tokenizer/tokenizer.joblib')


    # build model
    model: Model = get_model(vocab_size=vocab_size, max_len=max_len)

    # restore weights
    try:
        model_filename = f'{model_dir}/model.h5'
        model = tf.keras.models.load_model(model_filename)
    except:
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True
        )
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    

    # generate text until [SEP] token
    while True:
        word = input('시작 단어 입력(종료: Ctrl + C): ')
        print(generate_sentence(model, tokenizer, word, vocab_size))
        print('')
