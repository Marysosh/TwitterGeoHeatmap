import os

import pandas as pd

data_dir = "C:\\Users\\Stanislav\\Desktop\\vkr\\data\\"

def load_parsed(path):
    df = pd.DataFrame()
    for csv in os.listdir(path):
        df = df.append(pd.read_csv(path + csv, delimiter='\t',header=None))
    return df

if __name__ == "__main__":
    df = load_parsed(data_dir)
    print(df.head(10))

    compression_opts = {'method':'gzip',
                            'archive_name':'user_info.test.csv'}
    df.to_csv('user_info.test.gz', sep='	', index=False, header=False,
              compression='gzip')
