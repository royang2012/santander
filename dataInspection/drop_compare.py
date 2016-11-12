## Abandoned temporarily as the computatoin time is too long
import pandas as pd
from tqdm import tqdm
import numpy as np
train_df = pd.read_csv('../input/train_ver2.csv')
test_df = pd.read_csv('../input/test_ver2.csv')

train_df.set_index(['ncodpers'], inplace=True)
# test_df.set_index(['ncodpers'], inplace=True)

first_df = train_df.ix[15889]
#
# train_df['keep'] = pd.Series(np.zeros(train_df.shape[0]), index=train_df.index)
i = 1
temp = test_df['ncodpers'][i]
train_list = [first_df, train_df.ix[temp]]
for i in tqdm(range(2, 20)):
    temp_idx = test_df['ncodpers'][i]
    temp_df = train_df.ix[temp_idx]
    if temp_df.shape[0] == 17:
        # new_train_df = [new_train_df, train_df.ix[temp]]
        # train_list.append(temp_df)
        flat = True
    # train_df.set_value(temp, 'keep', train_df.ix[temp].shape[0] - 16)

new_train_df = pd.concat(train_list)
new_train_df.to_csv('./input/train17.csv')
