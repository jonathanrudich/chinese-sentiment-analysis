import pickle
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import array

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


def load_vocab():
    with open('data/processed/vocab.txt', 'r', encoding='utf8') as file:
        vocab = file.read()
        file.close()
        return vocab


###################################

# load processed df
file = open('data/processed/seg_df.pkl', 'rb')
processed_df = pickle.load(file)
file.close()

# load vocab
print('loading vocab...')
vocab = load_vocab()
vocab = vocab.split()
vocab = set(vocab)

# obtain train and test data from processed df
print('splitting data into train and test...')
train = random.sample(processed_df['seg_text'].tolist(), 7500)
test = random.sample(processed_df['seg_text'].tolist(), 816)

# create tokenizer and fit on documents
print('creating tokenizer...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train)

# use tokenizer to convert words to integer sequences
print('performing sequence encoding...')
encoded_train = tokenizer.texts_to_sequences(train)
encoded_test = tokenizer.texts_to_sequences(test)

# find longest sequence and pad training and test data
max_length = max([len(s) for s in train])

x_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
x_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')

# get half length of train and test length
half_train_length = int(round(len(train) / 2))
half_test_length = int(round(len(test) / 2))

# define train and test labels
y_train = array([0 for _ in range(half_train_length)] +
                [1 for _ in range(half_train_length)])
y_test = array([0 for _ in range(half_test_length)] +
               [1 for _ in range(half_test_length)])

# define vocab size (+1 for unknown words)
vocab_size = len(tokenizer.word_index) + 1

# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# save model
model.save('models/embedded_conv_1d')

# create dict of train and test data
data_dict = {'x_train': x_train, 'y_train': y_train,
             'x_test': x_test, 'y_test': y_test}

# serialize data dict
print('saving data dict to pickle...')
with open('data/processed/data_dict.pkl', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
