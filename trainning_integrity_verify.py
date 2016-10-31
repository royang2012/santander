from tqdm import tqdm
import pandas as pd

train_df = pd.read_csv('./input/train_ver2.csv')

test_df = pd.read_csv('./input/test_ver2.csv')

index_userId = 'ncodpers'
index_fetch_date = 'fecha_dato'
# check if all clients in test set is presented in training set

# # answer is yes
# result = [False] * test_df.shape[0]
# for i in range(0, test_df.shape[0]):
#     result[i] = test_df['ncodpers'][i] in train_df['ncodpers']
#
# print False in result
#
# sexo_list = []
# for i in tqdm(range(0, test_df.shape[0])):
#     index_i = test_df['ncodpers'][i]
#     # sexo_list.append(train_df[train_df.ix[[index_i].fecha_dato == '2015-01-28']].sexo)
#     sexo_list.append(train_df.xs((index_i, '2015-01-28'), level=('ncodpers', 'fecha_dato')).sexo.isnull())

train_df.set_index([index_userId], inplace=True)
train_df[index_userId]=train_df.index
test_df.set_index([index_userId], inplace=True)
test_df[index_userId]=test_df.index


# training data validation based test set
# validation_df=pd.merge(test_df, train_df, left_index=True, right_index=True, suffixes=('_test','_train'))
# validation_df.set_index([index_userId+'_train',index_fetch_date+'_train'], inplace=True)
# validation_df[validation_df.index=='2015-01-28' & validation_df['age_train'].isnull()].tail(5)
#
# validation_df.ix[[('2015-01-28', index_fetch_date+'_train')]].age_train.isnull()
