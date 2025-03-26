# Neural-Network-SMS-Text-Classifier

predict_message("your message here")


[0.92, 'spam']  # or [0.08, 'ham']


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters
vocab_size = 10000
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# Tokenize
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_data)

train_sequences = tokenizer.texts_to_sequences(train_data)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


import numpy as np

train_labels_final = np.array([1 if label == 'spam' else 0 for label in train_labels])
test_labels_final = np.array([1 if label == 'spam' else 0 for label in test_labels])


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

model = Sequential([
    Embedding(vocab_size, 16, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


num_epochs = 30
model.fit(train_padded, train_labels_final, epochs=num_epochs, validation_data=(test_padded, test_labels_final))


def predict_message(message):
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    pred = model.predict(padded)[0][0]
    label = 'spam' if pred > 0.5 else 'ham'
    return [float(pred), label]


predict_message("Congratulations! You won a free ticket to Bahamas. Text WIN to 55555.")


