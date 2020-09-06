import pandas as pd


def load_data_zh_file(dirname):

    data_zh_dir = 'data/raw/dataZH/dataZH/dataZH'

    it168_df = pd.DataFrame(columns=['sentiment', 'text'])

    with open(f'{data_zh_dir}/{dirname}/corpus.txt', 'r', encoding='utf8') as file:

        # skip product name
        line = file.readline()
        line = file.readline()

        while line:

            id = None
            rev_text = None
            sentiment = None

            line = line.strip()

            # indicates start of new review
            if line.isnumeric():

                # set id
                id = line

                line = file.readline()

                # parse review body
                if '<Rev_body>' in line:

                    line = file.readline()

                    # parse sentiment
                    if '<Sentiment>' in line and '</Sentiment>' in line:

                        # set sentiment
                        sentiment = line.replace(
                            '<Sentiment>', '').replace('</Sentiment>', '').strip()

                        line = file.readline()

                        # parse rev_text
                        if '<Rev_text>' in line and '</Rev_text>' in line:

                            # set rev_text
                            rev_text = line.replace(
                                '<Rev_text>', '').replace('</Rev_text>', '').rstrip()

            # create new dataframe row
            if id and rev_text and sentiment:
                df_row = pd.DataFrame(
                    {'sentiment': [sentiment], 'text': [rev_text]}, index=[id])
                # print(df_row)
                frames = [it168_df, df_row]
                it168_df = pd.concat(frames)
                # print(it168_df)

            line = file.readline()

    print(it168_df.head())
    print(it168_df.tail())

    # serialize and store in interim folder
    it168_df.to_pickle(f"data/interim/{dirname}_df.pkl")


# load_data_zh_file('it168test')
