import pandas as pd
from tqdm import tqdm

train_df = pd.read_csv('./input/sorted_train.csv')
train_df = train_df[:100]
train_list = []
last_idx = 15889
month_count = 0
for i in tqdm(range(1, 50)):
    if train_df.ix[i].ncodpers == last_idx:
        month_count += 1
        if month_count == 16:
            month_count = 0
            train_list.append(train_df.ix[i-16:i])
    else:
        month_count = 0
    last_idx = train_df.ix[i]
