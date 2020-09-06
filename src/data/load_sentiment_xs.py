import pandas as pd


def format_sentiment_labels(text):
    if text == 'positive':
        return 'pos'
    if text == 'negative':
        return 'neg'

#######################################


filename = 'data/raw/Chinese_conversation_sentiment-master/Chinese_conversation_sentiment-master/sentiment_XS_test.csv'

csv_fieldnames = ['sentiment', 'text']

print('reading csv to df...')
sentiment_xs_df = pd.read_csv(filename, encoding='utf8', names=csv_fieldnames)
sentiment_xs_df = sentiment_xs_df.drop(sentiment_xs_df.index[0])

print('formatting labels...')
sentiment_xs_df['sentiment'] = sentiment_xs_df['sentiment'].apply(
    lambda label: format_sentiment_labels(label))

print(sentiment_xs_df.head())
print(sentiment_xs_df.tail())

sentiment_xs_df.to_pickle("data/interim/sentiment_xs_df.pkl")
