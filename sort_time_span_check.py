import pandas as pd
from tqdm import tqdm

train_df = pd.read_csv('./input/sorted_train.csv')

train_list = []
last_idx = 15889
month_count = 0
for i in tqdm(range(1, train_df.shape[0])):
    if train_df.ix[i].ncodpers == last_idx:
        month_count += 1
        if month_count == 15:
            month_count = 0
            train_list.append(train_df.ix[i-16:i])
    else:
        month_count = 0
    last_idx = train_df.ix[i].ncodpers

train_17_df = pd.concat(train_list)
train_17_df.to_csv('./input/sorted_17_train.csv', index=False)
