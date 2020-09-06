from os import listdir
from load_it168 import load_data_zh_file

data_zh_dir = 'data/raw/dataZH/dataZH/dataZH'

dirname_list = [dirname for dirname in listdir(
    data_zh_dir) if '.txt' not in dirname]

# script to load reviews from all files in the data_ZH directory
for dirname in dirname_list:
    for filename in listdir(f'{data_zh_dir}/{dirname}'):
        with open(f'{data_zh_dir}/{dirname}/{filename}', 'r', encoding='utf8') as file:
            load_data_zh_file(dirname)
