import pandas as pd

# load sentiment corpus to dataframe

sentiment_types = {'pos', 'neg'}

senti_corp_df = pd.DataFrame(columns=['pos', 'neg'], index=range(0, 3000))

for sentiment_type in sentiment_types:

    # open dir of specified sentiment type
    dir = f'data/raw/ChnSentiCorp/ChnSentiCorp/utf-8/6000/6000/{sentiment_type}/'

    # iterate through all 6000 files
    for i in range(0, 3000):

        with open(f'{dir}{sentiment_type}.{i}.txt', encoding='utf8') as file:

            # add contents of file to dataframe
            senti_corp_df.loc[i][sentiment_type] = file.read()

print(senti_corp_df.head())
print(senti_corp_df.tail())

# serialize and store in interim folder
senti_corp_df.to_pickle("data/interim/senti_corp_df.pkl")
