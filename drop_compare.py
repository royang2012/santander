import pandas as pd
from tqdm import tqdm
import numpy as np
train_df = pd.read_csv('./input/train_ver2.csv')
test_df = pd.read_csv('./input/test_ver2.csv')

train_df.set_index(['ncodpers'], inplace=True)
# test_df.set_index(['ncodpers'], inplace=True)

# new_train_df = train_df.ix[15889]

train_df['keep'] = pd.Series(np.zeros(train_df.shape[0]), index=train_df.index)

for i in tqdm(range(1, test_df.shape[0])):
    temp = test_df['ncodpers'][i]
    # if train_df.ix[temp].shape[0] == 17:
    #     train_df.ix[temp].keep
    train_df.set_value(temp, 'keep', train_df.ix[temp].shape[0] - 16)

