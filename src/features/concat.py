import pickle
import pandas as pd
from os import listdir

## script to transfor and concatenate diffrent sentiment review dataframes ##

# open senti corp df file
file = open('data/interim/senti_corp_df.pkl', 'rb')

# get pickle
senti_corp_df = pickle.load(file)
file.close()

# open it168 df file
file = open('data/interim/it168_df.pkl', 'rb')

# get pickle
concat_df = pickle.load(file)
file.close()

data_zh_dir = 'data/raw/dataZH/dataZH/dataZH'

# concatenate all of data zh dfs to it168
dirname_list = [dirname for dirname in listdir(
    data_zh_dir) if '.txt' not in dirname]

for dirname in dirname_list:
    # open it168 df file
    file = open(f'data/interim/{dirname}_df.pkl', 'rb')

    # get pickle
    zh_df = pickle.load(file)
    file.close()

    # concatenate
    concat_df = pd.concat([concat_df, zh_df], ignore_index=True)


# transform senti corp df to match it168

#   pos    |    neg             sentiment | text
# 0  我愛她   我恨他   =>      0    neg       我恨他
# 1                           1    pos       我愛她

# split by column and concat
sentiment_types = ['pos', 'neg']

for sentiment_type in sentiment_types:
    for review in senti_corp_df[sentiment_type]:
        df_row = pd.DataFrame(
            {'sentiment': [sentiment_type], 'text': [review]})
        frames = [concat_df, df_row]
        concat_df = pd.concat(frames, ignore_index=True)

# add sentiment xs test df
print('adding sentiment xs test df...')
with open('data/interim/sentiment_xs_df.pkl', 'rb') as pickle_file:
    sentiment_xs_df = pickle.load(pickle_file)

# concatenate
concat_df = pd.concat([concat_df, sentiment_xs_df], ignore_index=True)


print(concat_df.tail())

# serialize and store in interim folder
concat_df.to_pickle("data/interim/concat_df.pkl")
