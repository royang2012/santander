# In this test training model, the training data is chosen to be a time series of 6 months history
# To save memory space, more columns are deleted. Also, all the numerical values are passed as int

import pandas as pd
import sqlite3 as sql
import xgboost as xgb
import numpy as np
# from dataPrepa import deleteColumns, convertNum, xgbDataGen
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)
#
# feature_df = pd.read_sql("select * from santander_train7 " +
#                             "where fecha_dato in ('2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', "
#                             "'2015-12-28', '2015-11-28') " +
#                             "order by ncodpers, fecha_dato DESC", santanderCon)
pred_month_df = pd.read_sql("select * from santander_train7 where " +
                            "fecha_dato = '2016-05-28' order by ncodpers", santanderCon)
last_month_df = pd.read_sql("select * from santander_train7 where " +
                            "fecha_dato = '2016-04-28' order by ncodpers", santanderCon)
secl_month_df = pd.read_sql("select * from santander_train7 where " +
                            "fecha_dato = '2016-03-28' order by ncodpers", santanderCon)

diff_df = last_month_df.ix[:, 25:49] - secl_month_df.ix[:, 25:49]
output_df = diff_df[diff_df == 1]
output_df = output_df.fillna(0)

# length = pre_month_df.shape[0]

# clean the data
deleteColumns(last_month_df)
deleteColumns(pred_month_df)

# force two columns to be numerical
convertNum(last_month_df)
convertNum(pred_month_df)

# find the data type for each column
dtype_list = last_month_df.dtypes
numerical_list = []
dummy_list = []
num_df = []
# separate numerical and non-numerical columns
for i in range(3, last_month_df.shape[1]):
    if(dtype_list[i] == 'object'):
        dummy_list.append(last_month_df.columns[i])
    else:
        numerical_list.append(last_month_df.columns[i])

# define the product to predict
predict_product = 'ind_ctop_fin_ult1'

# convert obejcts to One-hot-vector and combine the features
# For customer features, only one month is enough, so we use data from 2016-04-28
# note * indexes in numerical_list may change if total number of column changes!
numerical_list_feature = numerical_list[0: 8]

train_X = xgbDataGen(last_month_df, secl_month_df[predict_product], numerical_list_feature, dummy_list)
train_Y = output_df[predict_product].values

test_X = xgbDataGen(pred_month_df, last_month_df[predict_product], numerical_list_feature, dummy_list)
# test_Y = np.nan_to_num(added_df.ix[length*0.8:, 6].values)

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'binary:logitraw'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 5
param['silent'] = 1
param['nthread'] = 4
# param['num_class'] = 2

# watchlist = [ (xg_train,'train'), (xg_test, 'test')]
watchlist = [ (xg_train,'train')]
num_round = 10
bst = xgb.train(param, xg_train, num_round, watchlist);
# get prediction
pred = bst.predict( xg_test );


print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
print ('all zero guess error=%f' %(np.sum(test_Y)/test_Y.shape[0]))
