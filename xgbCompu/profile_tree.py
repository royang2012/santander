import pandas as pd
import sqlite3 as sql
import xgboost as xgb
import numpy as np
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)

pre_month_df = pd.read_sql("select new_product.* from santander_train new_product where " +
                            "new_product.fecha_dato = '2016-03-28' and new_product.ncodpers " +
                            "in (select ncodpers from santander_train where fecha_dato = '2016-04-28') " +
                            "order by ncodpers", santanderCon)

cur_month_df = pd.read_sql("select new_product.* from santander_train new_product where " +
                            "new_product.fecha_dato = '2016-04-28' and new_product.ncodpers " +
                            "in (select ncodpers from santander_train where fecha_dato = '2016-03-28') " +
                            "order by ncodpers", santanderCon)

length = pre_month_df.shape[0]

diff_df = pre_month_df.ix[:, 25:49] - cur_month_df.ix[:, 25:49]
added_df = diff_df[diff_df == 1]

# delete unwanted columns
del pre_month_df['fecha_alta']
del pre_month_df['ult_fec_cli_1t']
del pre_month_df['canal_entrada']
del pre_month_df['nomprov']

# force two columns to be numerical
pre_month_df['age'] = pd.to_numeric(pre_month_df['age'], errors='coerce')
pre_month_df['antiguedad'] = pd.to_numeric(pre_month_df['antiguedad'], errors='coerce')

# delete all product used info from the dataframe
product_column_list = np.arange(21,45)
pre_month_df.drop(pre_month_df.columns[product_column_list], axis=1, inplace=1)
# find the data type for each column
dtype_list = pre_month_df.dtypes
numerical_list = []
dummy_list = []
num_df = []
# separate numerical and non-numerical columns
for i in range(3, pre_month_df.shape[1]):
    if(dtype_list[i] == 'object'):
        dummy_list.append(pre_month_df.columns[i])
    else:
        numerical_list.append(pre_month_df.columns[i])

# convert obejcts to One-hot-vector and combine the features
num_df = pre_month_df[numerical_list]
num_df = pd.concat([num_df, pd.get_dummies(pre_month_df[dummy_list])], axis=1)

# use the first 80% of customers for training and the last 20% for validation
train_data = num_df.ix[0:length*0.8, 0:].values
test_data = num_df.ix[length*0.8:, 0:].values

train_X = train_data
train_Y = np.nan_to_num(added_df.ix[0:length*0.8, 6].values)

test_X = test_data
test_Y = np.nan_to_num(added_df.ix[length*0.8:, 6].values)

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
# param['objective'] = 'multi:softmax'
param['objective'] = 'binary:logistic'

# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 5
param['silent'] = 1
param['nthread'] = 4
# param['num_class'] = 2

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 20
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction
pred = bst.predict( xg_test );


print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
print ('all zero guess error=%f' %(np.sum(test_Y)/test_Y.shape[0]))
