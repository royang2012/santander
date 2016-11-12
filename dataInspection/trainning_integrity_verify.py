from tqdm import tqdm
import pandas as pd

train_df = pd.read_csv('../input/train_ver2.csv')

test_df = pd.read_csv('../input/test_ver2.csv')

# index_userId = 'ncodpers'
# index_fetch_date = 'fecha_dato'
#***************************************************************#
# check if all clients in test set is presented in training set
#***************************************************************#
# # answer is yes
result = [False] * test_df.shape[0]
ncodpers = train_df['ncodpers']
for i in tqdm(range(0, test_df.shape[0])):
    result[i] = test_df.ix[i]['ncodpers'] in ncodpers

print False in result

#********************************************#
# Some abandoned test based on sexo column
#********************************************#
# sexo_list = []
# for i in tqdm(range(0, test_df.shape[0])):
#     index_i = test_df['ncodpers'][i]
#     # sexo_list.append(train_df[train_df.ix[[index_i].fecha_dato == '2015-01-28']].sexo)
#     sexo_list.append(train_df.xs((index_i, '2015-01-28'), level=('ncodpers', 'fecha_dato')).sexo.isnull())

# train_df.set_index([index_userId], inplace=True)
# train_df[index_userId]=train_df.index
# test_df.set_index([index_userId], inplace=True)
# test_df[index_userId]=test_df.index

  #********************************************#
# training data validation based test set
#********************************************#
# validation_df=pd.merge(test_df, train_df, left_index=True, right_index=True, suffixes=('_test','_train'))
# validation_df.set_index([index_userId+'_train',index_fetch_date+'_train'], inplace=True)
# validation_df[validation_df.index=='2015-01-28' & validation_df['age_train'].isnull()].tail(5)
#
# validation_df.ix[[('2015-01-28', index_fetch_date+'_train')]].age_train.isnull()

#**********************************************#
# How many customers that is presented in the test set has full 17 month history?
# only 408513, 54.6%
#**********************************************#
# train_df = pd.read_csv('./input/sorted_17_train.csv')
# test_df = pd.read_csv('./input/test_ver2.csv')
#
# unique_idx = train_df.ncodpers.unique()
# unique_idx_test = test_df.ncodpers.unique()
# result = [False] * unique_idx.shape[0]
# for i in tqdm(range(0, unique_idx.shape[0])):
#     result[i] = unique_idx[i] in unique_idx_test
# print 'uniques in train'
# print unique_idx.shape[0]
# print 'uniques in test'
# print unique_idx_test.shape[0]
# print 'both'
# print result.count(True)
