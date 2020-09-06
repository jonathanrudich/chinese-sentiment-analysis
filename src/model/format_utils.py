import pkuseg
import numpy as np
import torch
from string import punctuation
from hanziconv import HanziConv
import pickle

# get train on gpu


def get_train_on_gpu():

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()

    if(train_on_gpu):
        print('Training on GPU.')
        return True
    else:
        print('No GPU available, training on CPU.')
        return False


def remove_stopwords(seg_text):
    # load stopwords simple df
    file = open('data/interim/stopwords_simpl_df.pkl', 'rb')
    stopwords_simpl_df = pickle.load(file)
    file.close()

    no_stopwords = [
        word for word in seg_text if
        word not in stopwords_simpl_df['stopword'].tolist()
        and not word.isnumeric() and word not in punctuation]
    return no_stopwords

# create vocab to int dict


def get_vocab_to_int_dict():

    # load vocab
    vocab = load_vocab()

    # create vocab to int dict
    #print('creating vocab to int dict...')
    vocab_sorted = sorted(vocab, reverse=True)
    vocab_to_int_dict = {word: ii for ii, word in enumerate(vocab_sorted, 1)}
    return vocab_to_int_dict


def load_vocab():
    with open('data/processed/vocab.txt', 'r', encoding='utf8') as file:
        vocab = file.read()
        file.close()
        vocab = vocab.split()
        return vocab

# method to format sample text to be evaluated by model


def format_text(text, no_stopwords):

    print('removing stopwords') if no_stopwords == 'yes' else print(
        'not removing stopwords')

    # convert to simplified
    text = HanziConv.toSimplified(text)

    # remove punctuation
    no_punc = ''.join([c for c in text if c not in punctuation]
                      ) if no_stopwords == 'yes' else text

    # initialize pkuseg
    seg = pkuseg.pkuseg()

    # cut text
    seg_text = seg.cut(no_punc)

    # remove stopwords
    no_stopwords = remove_stopwords(
        seg_text) if no_stopwords == 'yes' else seg_text

    # get tokens
    tokens = embed_text(no_stopwords)

    # pad reviews with length 200
    padded_tokens = pad_reviews(tokens, 200)

    # create feature tensor
    feature_tensor = torch.LongTensor([padded_tokens])

    #print('Formatted text')
    #print(f'Feature Tensor: {feature_tensor}')
    #print(f'Size: {feature_tensor.size()}')

    return feature_tensor


# perform word embedding using int to word dict


def embed_text(text):
    vocab_to_int_dict = get_vocab_to_int_dict()
    text = [vocab_to_int_dict[word]
            for word in text if word in vocab_to_int_dict]
    return text

# pad reviews with zeros


def pad_reviews(review, max_len):

    padded = np.zeros(max_len, dtype=int)

    for i in range(0, 200):
        if i < len(review):
            padded[i] = review[i]

    return padded
