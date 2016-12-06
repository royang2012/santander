import xgboost as xgb
import pandas as pd
import sqlite3 as sql
from tqdm import tqdm
import numpy as np
import random
from matplotlib import pyplot

def genRandClass():
    position_list = list(range(16))
    hyper_class = np.zeros([4, 16])
    for i in range(0, 4):
        random.shuffle(position_list)
        hyper_class[i, position_list[0: 4]] = 1
        del position_list[0: 3]
    return hyper_class

def genClassLabel(input_df, class_set):
    picked_class = np.zeros(input_df.shape[0])
    for i in range(0, input_df.shape[0]):
        # compute the score of each class by dot product
        class_score = np.dot(input_df.ix[i].values, class_set.transpose())
        # the class is assigned where the first '1' lies
        picked_class[i] = np.argmax(class_score)
    return picked_class

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
    del input_df['ind_cder_fin_ult1']
    del input_df['ind_deco_fin_ult1']
    del input_df['ind_deme_fin_ult1']
    del input_df['ind_hip_fin_ult1']
    del input_df['ind_pres_fin_ult1']
    del input_df['ind_viv_fin_ult1']

def convertNum(intput_df):
    intput_df['age'] = pd.to_numeric(intput_df['age'], errors='coerce')
    intput_df['antiguedad'] = pd.to_numeric(intput_df['antiguedad'], errors='coerce')

product_num = 16
info_num = 16
# # # ********************************************* # # #
# # # read data from database
# # # ********************************************* # # #
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)
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
# data
train_ary = np.zeros([train_fea_df.shape[0]/6, product_num * 6])
vali_ary = np.zeros([vali_fea_df.shape[0]/6, product_num * 6])
for i in range(0, train_fea_df.shape[0]/6):
    train_ary[i, :] = train_fea_df.ix[6*i: 6*i+5, info_num:].values.\
        reshape(product_num * 6)
for i in range(0, vali_fea_df.shape[0]/6):
    vali_ary[i, :] = vali_fea_df.ix[6*i: 6*i+5, info_num:].values.\
        reshape(product_num * 6)
# re-sample the feature data frame at every 6 rows
train_fea_df = train_fea_df.iloc[::6]
vali_fea_df = vali_fea_df.iloc[::6]
# find the data type for each column
# note that here training and validation set share the same input stucture,
# so we use one label list for both of them
dtype_list = train_fea_df.dtypes
train_numerical_list = []
train_dummy_list = []
# separate numerical and non-numerical columns
for i in range(2, info_num): # * the number might vary!
    if(dtype_list[i] == 'object'):
        train_dummy_list.append(train_fea_df.columns[i])
    else:
        train_numerical_list.append(train_fea_df.columns[i])
train_num_ary = train_fea_df[train_numerical_list].values
# train_dum_ary = pd.get_dummies(train_fea_df[train_dummy_list]).values
vali_num_ary = vali_fea_df[train_numerical_list].values
# vali_dum_ary = pd.get_dummies(vali_fea_df[train_dummy_list]).values
dummy_df = pd.get_dummies(pd.concat(
    [train_fea_df[train_dummy_list], vali_fea_df[train_dummy_list]], axis=0))
train_dum_ary = dummy_df[:train_fea_df.shape[0]]
vali_dum_ary = dummy_df[train_fea_df.shape[0]:]
# concatenate all features(products, info) into one array
train_ary = np.concatenate((train_ary, train_num_ary, train_dum_ary), axis=1)
vali_ary = np.concatenate((vali_ary, vali_num_ary, vali_dum_ary), axis=1)
# # # ********************************************* # # #
# # # set the tree parameters
# # # ********************************************* # # #
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.12
param['max_depth'] = 7
param['silent'] = 1
# param['nthread'] = 4
param['num_class'] = 4
# # # ********************************************* # # #
# # # get the hyper-classes labels
# # # ********************************************* # # #
num_tree = 1
p_pred = np.zeros(vali_out_df.shape)
hyper_class = genRandClass()
train_labels = genClassLabel(train_out_df, hyper_class)
# train a tree
xg_train = xgb.DMatrix(train_ary, label=train_labels)
watchlist = [ (xg_train,'train')]
num_round = 500
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction
xg_test = xgb.DMatrix(vali_ary)
hc_pred = bst.predict( xg_test )
    # compute the real products added according to the prediction
    # p_pred += hyper_class[tuple(hc_pred.astype(int)), :]

xgb.plot_importance(bst)
pyplot.show()
# final_prediction = np.zeros(vali_out_df.shape)
# for i in range(0, vali_out_df.shape[0]):
#     seven_positions = np.argpartition(np.array(p_pred[i]), -7)[-7:]
#     final_prediction[i, seven_positions] = 1
# np.savetxt("../input/pred_1130.csv", final_prediction, delimiter=",")
# total_customers = 897377
# score = 0.0
# for i in range(0, vali_out_df.shape[0]):
#     real_array = vali_out_df.ix[i].values
#     s = np.dot(real_array, final_prediction[i])
#     if s != 0:
#         score += s/sum(vali_out_df.ix[i])
# score /= total_customers
# print score
#
# full_prediction = np.ones(vali_out_df.shape[1])
# score = 0.0
# for i in range(0, vali_out_df.shape[0]):
#     real_array = vali_out_df.ix[i].values
#     s = np.dot(real_array, full_prediction)
#     if s != 0:
#         score += s/sum(vali_out_df.ix[i])
# score /= total_customers
# print score
