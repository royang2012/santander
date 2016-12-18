import pandas as pd
import numpy as np
from tqdm import tqdm
test_df = pd.read_csv('./input/test_ver2.csv')
train_df = pd.read_csv('./input/train_ver2.csv')

test_df.sort_values(['ncodpers'], inplace=True)
test_df.reset_index(inplace=True)
del test_df['index']
del test_df['level_0']
# train_df.sort_values(['ncodpers', 'fecha_dato'], ascending=[True, False])
df0528 = train_df[train_df['fecha_dato']=='2016-05-28']
df0428 = train_df[train_df['fecha_dato']=='2016-04-28']

# his_df = pd.concat([df0428, df0528])
# his_df.sort_values(['ncodpers', 'fecha_dato'], ascending=[True, False], inplace =True)
# his_df.reset_index(inplace=True)
# j = 0
# for i in tqdm(range(0, test_df.shape[0])):
#     if test_df.ix[i, 'ncodpers'] == his_df.ix[j, 'ncodpers']:
#         if his_df.ix[j, 'ncodpers'] == his_df.ix[j, 'ncodpers']
merged_df1 = pd.merge(test_df, df0428, how='left', on='ncodpers')
merged_df2 = pd.merge(test_df, df0528, how='left', on='ncodpers')
his_df = pd.concat([merged_df1, merged_df2])
his_df.sort_values(['ncodpers', 'fecha_dato'], ascending=[True, False], inplace =True)
his_df.reset_index(inplace=True)
his_df.to_csv('./input/pred_his2.csv')
