from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense


def get_model(vocab_size, max_len, embedding_size=128, hidden_size=128, dropout_rate=0.3):

    # layers
    inputs = Input(shape=(max_len,))
    embedding = Embedding(vocab_size, embedding_size)
    lstm_1 = Bidirectional(LSTM(hidden_size, dropout=dropout_rate, return_sequences=True))
    lstm_2 = Bidirectional(LSTM(hidden_size, dropout=dropout_rate, return_sequences=True))
    lstm_3 = Bidirectional(LSTM(hidden_size, dropout=dropout_rate))
    fc = Dense(vocab_size)

    # forward pass
    hidden_emb = embedding(inputs)
    hidden_lstm_1 = lstm_1(hidden_emb)
    hidden_lstm_2 = lstm_2(hidden_lstm_1)
    hidden_lstm_3 = lstm_3(hidden_lstm_2)
    outputs = fc(hidden_lstm_3)

    # model
    model = Model(inputs=inputs, outputs=outputs)

    return model
