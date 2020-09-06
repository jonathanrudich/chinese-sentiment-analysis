import pickle
import pkuseg
from collections import Counter
from string import punctuation
import numpy as np
import sys

# perform processing on segmented text


def process_seg_text(seg_text, vocab, no_stopwords):
    words = remove_stopwords(seg_text) if no_stopwords == 'yes' else seg_text
    add_to_vocab(vocab, words)
    return words


# remove all stopwords, numbers and punctuation from text


def remove_stopwords(seg_text):
    no_stopwords = [
        word for word in seg_text if
        word not in stopwords_simpl_df['stopword'].tolist()
        and not word.isnumeric() and word not in punctuation]
    return no_stopwords

# add words to vocab


def add_to_vocab(vocab, words):
    vocab.update(words)

# remove words that don't meet min occurrence, for this case set to 3


def min_occurrence(vocab, no_stopwords):
    min_occurrence_count = 3
    vocab_words = [word for word,
                   count in vocab.items() if count >= min_occurrence_count] if no_stopwords == 'yes' else [word for word, count in vocab.items()]
    return vocab_words


# save vocab to text file


def save_vocab_to_file(vocab_words):
    data = '\n'.join(vocab_words)
    file = open('data/processed/vocab.txt', 'w', encoding='utf8')
    file.write(data)
    file.close()

###################################################


# if yes, remove stopwords, else do not
no_stopwords = sys.argv[0]
print('will remove stopwords') if no_stopwords == 'yes' else print(
    'will not remove stopwords')

# open concatenated df file and get pickle
file = open('data/interim/concat_df.pkl', 'rb')
concat_df = pickle.load(file)
file.close()

# apply segmentation to all text fields in dataframe and add results to new column
print('performing word segmentation...')
seg = pkuseg.pkuseg()
concat_df['seg_text'] = concat_df['text'].apply(lambda text: seg.cut(text))
print(concat_df.tail())

# load stopwords simple df
file = open('data/interim/stopwords_simpl_df.pkl', 'rb')
stopwords_simpl_df = pickle.load(file)
file.close()

# initialize vocab counter
vocab = Counter()

# remove all stopwords from segmented df and add words to vocab
print('removing stopwords and creating vocab...')
concat_df['seg_text'] = concat_df['seg_text'].apply(
    lambda seg_text: process_seg_text(seg_text, vocab, no_stopwords))
print(concat_df.tail())

# remove words from vocab that do not meet min occurrence
print(f'before min occurrence: {len(vocab)}')
vocab_words = min_occurrence(vocab, no_stopwords)
print(f'after min occurrence: {len(vocab_words)}')

# remove these words from df
print('removing low frequency words from df...')
concat_df['seg_text'] = concat_df['seg_text'].apply(
    lambda text: [word for word in text if word in vocab_words])

# save vocab to text file
print('saving vocab and df to file...')
save_vocab_to_file(vocab_words)

# save segmented df to pickle
torch_rnn_format_df = concat_df
torch_rnn_format_df.to_pickle('data/processed/torch_rnn_processed_df.pkl')
