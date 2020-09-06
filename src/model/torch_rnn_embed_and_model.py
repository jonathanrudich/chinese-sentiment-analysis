import pickle
import numpy as np
import statistics
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_rnn_model import SentimentRNN

# save vocab to text file


def load_vocab():
    with open('data/processed/vocab.txt', 'r', encoding='utf8') as file:
        vocab = file.read()
        file.close()
        vocab = vocab.split()
        return vocab

# perform word embedding using int to word dict


def embed_text(text):
    text = [vocab_to_int_dict[word] for word in text]
    return text

# pad reviews with zeros


def pad_reviews(review, max_len):

    padded = np.zeros(max_len, dtype=int)

    for i in range(0, 200):
        if i < len(review):
            padded[i] = review[i]

    return padded


def split_dfs(processed_df):

    # get random sample for train df
    train_df, test_validate_df = train_test_split(processed_df, test_size=.1)
    print(f'number of elements in train df: {len(train_df)}')
    print(f'number of elements in test validate df: {len(test_validate_df)}')

    # split test validate df into train and test
    test_validate_dfs = np.array_split(test_validate_df, 2)
    test_df = pd.DataFrame(test_validate_dfs[0])
    validate_df = pd.DataFrame(test_validate_dfs[1])
    print(f'number of elements in test df: {len(test_df)}')
    print(f'number of elements in validate df: {len(validate_df)}')
    return train_df, test_df, validate_df


#################################################


# load processed df
file = open('data/processed/torch_rnn_processed_df.pkl', 'rb')
processed_df = pickle.load(file)
file.close()

# load vocab
vocab = load_vocab()

# create dict that maps vocab to int, save to file
print('creating vocab to int dict...')
vocab_sorted = sorted(vocab, reverse=True)
vocab_to_int_dict = {word: ii for ii, word in enumerate(vocab_sorted, 1)}

# tokenize reviews
print('tokenizing reviews and adding to df...')
processed_df['embedded'] = processed_df['seg_text'].apply(
    lambda text: [vocab_to_int_dict[word] for word in text])

# find max and average length review
lengths_array = [len(s) for s in processed_df['embedded'].tolist()]
max_len = max([len(s) for s in processed_df['embedded'].tolist()])
print(sum(lengths_array))
print(len(lengths_array))
avg_len = sum(lengths_array) / len(lengths_array)
print(f'max length review: {max_len}')
print(f'avg length review: {max_len}')

# set arbitrary length of reviews
review_len = 200

# pad with zeros
processed_df['embedded'] = processed_df['embedded'].apply(
    lambda review: pad_reviews(review, review_len))

# add encoded labels column
processed_df['encoded_labels'] = processed_df['sentiment'].apply(
    lambda label: 1 if label == 'pos' else 0)

print(processed_df['embedded'].loc[0])

# split into train, test and validation sets
print('splitting data into train, test and validate sets...')
train_df, test_df, validate_df = split_dfs(processed_df)

train_x = train_df['embedded'].tolist()
train_y = train_df['encoded_labels'].tolist()

test_x = test_df['embedded'].tolist()
test_y = test_df['encoded_labels'].tolist()

validate_x = validate_df['embedded'].tolist()
validate_y = validate_df['encoded_labels'].tolist()

# set batch size for dataloader
batch_size = 50

# create dataloaders and perform batching
train_data = TensorDataset(torch.LongTensor(
    train_x), torch.LongTensor(train_y))
test_data = TensorDataset(torch.LongTensor(test_x), torch.LongTensor(test_y))
validate_data = TensorDataset(torch.LongTensor(
    validate_x), torch.LongTensor(validate_y))

train_loader = DataLoader(train_data, shuffle=True,
                          batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True,
                         batch_size=batch_size, drop_last=True)
validate_loader = DataLoader(
    validate_data, shuffle=True, batch_size=batch_size, drop_last=True)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size())  # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size())  # batch_size
print('Sample label: \n', sample_y)

# instantiate the model with hyperparams
# length of the vocab mapping +1 for zero padding
vocab_size = len(vocab_to_int_dict) + 1
output_size = 1  # output size
embedding_dim = review_len  # length of each review
hidden_dim = 512
n_layers = 3

print('creating model...')
net = SentimentRNN(vocab_size, output_size,
                   embedding_dim, hidden_dim, n_layers)
print(net)

print('saving data to file...')
torch.save(train_loader, 'data/processed/torch_rnn_train.loader')
torch.save(test_loader, 'data/processed/torch_rnn_test.loader')
torch.save(validate_loader, 'data/processed/torch_rnn_validate.loader')

print('saving model to file...')
torch.save(net, 'models/torch_rnn.model')
