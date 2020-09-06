import pandas as pd

# create stopwords simpl df
stopwords_simpl_df = pd.DataFrame(columns=['stopword'])

# read file into df
with open('data/raw/stopwords/stopwords_simpl.txt', 'r', encoding='utf8') as file:

    line = file.readline()
    while line:
        df_row = pd.DataFrame(
            {'stopword': [line.strip()]})
        # print(df_row)
        frames = [stopwords_simpl_df, df_row]
        stopwords_simpl_df = pd.concat(frames, ignore_index=True)
        line = file.readline()

print(stopwords_simpl_df.tail())

# serialize to pickle
stopwords_simpl_df.to_pickle("data/interim/stopwords_simpl_df.pkl")
