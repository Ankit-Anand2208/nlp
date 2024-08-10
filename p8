import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import LambdaCallback
import random
import sys
with open('text_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()
chars = sorted(list(set(text)))
char_indices = {char: i for i, char in enumerate(chars)}
indices_char = {i: char for i, char in enumerate(chars)}
max_len = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - max_len, step):
    sentences.append(text[i: i + max_len])
    next_chars.append(text[i + max_len])
x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool_)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool_)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
model = Sequential()
model.add(LSTM(128, input_shape=(max_len, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
def generate_text(seed_text, temperature=0.5, generated_text_length=400):
    generated_text = seed_text.lower()
    for i in range(generated_text_length):
        x_pred = np.zeros((1, max_len, len(chars)))
    for t, char in enumerate(seed_text):
        x_pred[0, t, char_indices[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]
        generated_text += next_char
        seed_text = seed_text[1:] + next_char
    return generated_text
def on_epoch_end(epoch, _):
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    start_index = random.randint(0, len(text) - max_len - 1)
    for temperature in [0.2, 0.5, 1.0]:
        seed_text = text[start_index: start_index + max_len]
        generated_text = generate_text(seed_text, temperature)
        print('----- Temperature:', temperature)
        print(seed_text + generated_text)
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x, y,batch_size=128,epochs=30,callbacks=[print_callback])
