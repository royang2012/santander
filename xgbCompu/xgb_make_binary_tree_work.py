# from previous script, seems binary tress does not work for all products.
# to be detailed, the eval-auc score does not improve with trees added
# this script will pinpoint on those products and try to make them work.
# name of products where train failed: ind_valo_fin_ult1,ind_fond_fin_ult1,ind_plan_fin_ult1
# name of products where train stops early: ind_tjcr_fin_ult1, ind_dela_fin_ult1,
#                                              ind_ctpp_fin_ult1, ind_hip_fin_ult1
#
# columns list:
# # [u'fecha_dato', u'ncodpers', u'ind_empleado', u'sexo', u'age',
# #    u'ind_nuevo', u'antiguedad', u'indrel', u'indrel_1mes', u'tiprel_1mes',
# #    u'indresi', u'conyuemp', u'cod_prov', u'ind_actividad_cliente',
# #    u'renta', u'segmento', u'ind_cco_fin_ult1', u'ind_cder_fin_ult1',
# #    u'ind_cno_fin_ult1', u'ind_ctju_fin_ult1', u'ind_ctma_fin_ult1',
# #    u'ind_ctop_fin_ult1', u'ind_ctpp_fin_ult1', u'ind_dela_fin_ult1',
# #    u'ind_ecue_fin_ult1', u'ind_fond_fin_ult1', u'ind_hip_fin_ult1',
# #    u'ind_plan_fin_ult1', u'ind_pres_fin_ult1', u'ind_reca_fin_ult1',
# #    u'ind_tjcr_fin_ult1', u'ind_valo_fin_ult1', u'ind_viv_fin_ult1',
# #    u'ind_nomina_ult1', u'ind_nom_pens_ult1', u'ind_recibo_ult1']
#
# best number of rounds for all products:
# # [ 108.   11.  108.   13.   51.   14.   42.    2.   11.   70.    1.    7.
# #    8.   30.  115.   10.    1.  104.  106.   92.]
import xgboost as xgb
import pandas as pd
import sqlite3 as sql
from tqdm import tqdm
import numpy as np
import random


def deleteFeatures(input_df):
    # delete unwanted columns
    del input_df['fecha_alta']
    del input_df['ult_fec_cli_1t']
    del input_df['canal_entrada']
    del input_df['nomprov']
    # delete more data!
    del input_df['indext']
    del input_df['indfall']
    del input_df['tipodom']

def deleteProducts(input_df):
    del input_df['ind_ahor_fin_ult1']
    del input_df['ind_aval_fin_ult1']
    del input_df['ind_deco_fin_ult1']
    del input_df['ind_deme_fin_ult1']

def convertNum(intput_df):
    intput_df['age'] = pd.to_numeric(intput_df['age'], errors='coerce')
    intput_df['antiguedad'] = pd.to_numeric(intput_df['antiguedad'], errors='coerce')

product_num = 20
info_num = 16
# # # ********************************************* # # #
# # # read data from database
# # # ********************************************* # # #
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)

# deal with the training part
select_train_feature = "select * from santander_feature6_train"
train_fea_df = pd.read_sql(select_train_feature, santanderCon)
select_train_out = "select * from santander_out_train"
train_out_df = pd.read_sql(select_train_out, santanderCon)
select_vali_feature = "select * from santander_feature6_vali"
vali_fea_df = pd.read_sql(select_vali_feature, santanderCon)
select_vali_out = "select * from santander_out_vali"
vali_out_df = pd.read_sql(select_vali_out, santanderCon)
# some raw pre-processing of the data
del train_fea_df['level_0']
del train_fea_df['index']
deleteFeatures(train_fea_df)
deleteProducts(train_fea_df)
del vali_fea_df['level_0']
del vali_fea_df['index']
deleteFeatures(vali_fea_df)
deleteProducts(vali_fea_df)
del train_out_df['index']
deleteProducts(train_out_df)
del vali_out_df['index']
deleteProducts(vali_out_df)

del train_fea_df['pais_residencia']
del vali_fea_df['pais_residencia']

convertNum(train_fea_df)
convertNum(vali_fea_df)

# # # ********************************************* # # #
# # # prepare the data from xgb trees
# # # ********************************************* # # #
# delete the country as it generates too many hot vectors
# organize the data so that each feature row contains all info and 6 months of product
# data. + 1 is for the total number of products column

# re-sample the feature data frame at every 6 rows
train_fea_df6 = train_fea_df.iloc[::6].reset_index(drop=True)
vali_fea_df6 = vali_fea_df.iloc[::6].reset_index(drop=True)
# find the data type for each column
# note that here training and validation set share the same input stucture,
# so we use one label list for both of them
dtype_list = train_fea_df.dtypes
train_numerical_list = []
train_dummy_list = []
# separate numerical and non-numerical columns
# for i in range(2, info_num): # * the number might vary!
#     if(dtype_list[i] == 'object'):
#         train_dummy_list.append(train_fea_df.columns[i])
#     else:
#         train_numerical_list.append(train_fea_df.columns[i])
train_dummy_list = ['fecha_dato','ind_empleado',
                    'sexo','indrel_1mes','tiprel_1mes',
                    'indresi','conyuemp','segmento']
train_numerical_list = ['age','antiguedad','indrel','ind_actividad_cliente',
                        'renta', 'cod_prov','ind_nuevo']
train_num_ary = train_fea_df6[train_numerical_list].values
vali_num_ary = vali_fea_df6[train_numerical_list].values
total_df = pd.concat([train_fea_df6[train_dummy_list],
                      vali_fea_df6[train_dummy_list]])
dummy_df = pd.get_dummies(total_df[train_dummy_list])
train_dum_df = dummy_df[0:train_fea_df6.shape[0]]
vali_dum_df = dummy_df[train_fea_df6.shape[0]:]

train_dum_ary = train_dum_df.values
# if some value never appear in the training set, then delete it from the test set
# pred_dum_df_selected = pred_dum_df[train_dum_df.columns]
vali_dum_ary = vali_dum_df.values
# compute the total number of products for each month
total_train_prod = np.sum(train_fea_df.ix[:, info_num:].values, axis=1)
total_vali_prod = np.sum(vali_fea_df.ix[:, info_num:].values, axis=1)

# # # ********************************************* # # #
# # # set the tree parameters
# # # ********************************************* # # #
param = {}
# use softmax multi-class classification
param['objective'] = 'binary:logitraw'
# param['objective'] = 'reg:logistic'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 12
param['silent'] = 1
# param['nthread'] = 4
# param['num_class'] = 4
param['colsample_bytree'] = 0.8
param['subsample'] = 0.8
param['seed'] = 100
param['min_child_weight'] = 1.1
param['gamma'] = 10
num_round = 350
p_pred = np.zeros(vali_fea_df.shape[0]/6)
train_prod_ary = np.zeros([train_fea_df.shape[0]/6, 12])
pred_prod_ary = np.zeros([vali_fea_df.shape[0]/6, 12])
best_num_rounds = 0
# # # ********************************************* # # #
# # # focus on one binary tree
# # # ********************************************* # # #
# get the name of the product to be predicted
product_name = train_fea_df.columns[info_num + 10]
# get the 6-month history of that product and total # of products
train_prod_info = np.stack((train_fea_df[product_name].values,
    total_train_prod)).transpose()
vali_prod_info = np.stack((vali_fea_df[product_name].values,
    total_vali_prod)).transpose()
for i in range(0, train_fea_df.shape[0]/6):
    train_prod_ary[i] = train_prod_info[6*i: 6*i+6, :].reshape(12)
for i in range(0, vali_fea_df.shape[0]/6):
    pred_prod_ary[i] = vali_prod_info[6*i: 6*i+6, :].reshape(12)

# concatenate all features(products, info) into one array
train_ary = np.concatenate((train_prod_ary, train_num_ary, train_dum_ary), axis=1)
vali_ary = np.concatenate((pred_prod_ary, vali_num_ary, vali_dum_ary), axis=1)
# train_ary = np.concatenate((train_num_ary[:, 0:1], train_num_ary[:, 4:5]), axis=1)
# vali_ary = np.concatenate((vali_num_ary[:, 0:1], vali_num_ary[:, 4:5]), axis=1)
train_labels = train_out_df[product_name].values
vali_labels = vali_out_df[product_name].values
# train a tree
xg_train = xgb.DMatrix(train_ary, label=train_labels)
xg_test = xgb.DMatrix(vali_ary, label=vali_labels)
watchlist = [(xg_train, 'train'), (xg_test, 'eval')]

bst = xgb.train(param, xg_train, num_round, watchlist,
                early_stopping_rounds=100)
# get prediction
best_num_rounds = bst.best_ntree_limit
hc_pred = bst.predict(xg_test, ntree_limit=bst.best_ntree_limit)
# compute the real products added according to the prediction
p_pred += hc_pred
print product_name

# analyzing product 'ind_fond_fin_ult1'. The most important 5 features are:
# income, antiguedad, age, code_prov, product_history on last month
