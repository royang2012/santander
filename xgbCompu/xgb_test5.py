import xgboost as xgb
import pandas as pd
import sqlite3 as sql
from tqdm import tqdm
import numpy as np
import random


def genRandClass():
    position_list = list(range(16))
    hyper_class = np.zeros([4, 16])
    for i in range(0, 4):
        random.shuffle(position_list)
        hyper_class[i, position_list[0: 4]] = 1
        del position_list[0: 4]
    return hyper_class

def genClassLabel(input_df, class_set):
    picked_class = np.zeros(input_df.shape[0])
    for i in range(0, input_df.shape[0]):
        # compute the score of each class by dot product
        class_score = np.dot(input_df.ix[i].values  , class_set.transpose())
        # the class is assigned where the first '1' lies
        picked_class[i] = np.argmax(class_score)
    return picked_class

def genHistoryLabel(input_df, class_set):
    history_ary = np.dot(input_df.values, class_set.transpose)
    return history_ary

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
class_num = 4
# # # ********************************************* # # #
# # # read data from database
# # # ********************************************* # # #
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)

# deal with the prediction part
pred_fea_df = pd.read_csv("../input/pred6.csv",
                          dtype={'ind_ahor_fin_ult1':int, 'ind_aval_fin_ult1': int,
                                 'ind_cco_fin_ult1':int, 'ind_cder_fin_ult1':int,
                                 'ind_cno_fin_ult1':int, 'ind_ctju_fin_ult1':int,
                                 'ind_ctma_fin_ult1':int, 'ind_ctop_fin_ult1':int,
                                 'ind_ctpp_fin_ult1':int, 'ind_deco_fin_ult1':int,
                                 'ind_deme_fin_ult1':int, 'ind_dela_fin_ult1':int,
                                 'ind_ecue_fin_ult1':int, 'ind_fond_fin_ult1':int,
                                 'ind_hip_fin_ult1':int, 'ind_plan_fin_ult1':int,
                                 'ind_pres_fin_ult1':int, 'ind_reca_fin_ult1':int,
                                 'ind_tjcr_fin_ult1':int, 'ind_valo_fin_ult1':int,
                                 'ind_viv_fin_ult1':int, 'ind_nomina_ult1': int,
                                 'ind_nom_pens_ult1': int, 'ind_recibo_ult1': int
                                 })
del pred_fea_df['Unnamed: 0']
del pred_fea_df['pais_residencia']
deleteProducts(pred_fea_df)
deleteFeatures(pred_fea_df)
convertNum(pred_fea_df)
pred_fea_df.set_value(pred_fea_df['indrel_1mes']==1, 'indrel_1mes', '1.0')
pred_fea_df.set_value(pred_fea_df['indrel_1mes']==2, 'indrel_1mes', '2.0')
pred_fea_df.set_value(pred_fea_df['indrel_1mes']==3, 'indrel_1mes', '3.0')
pred_fea_df.set_value(pred_fea_df['indrel_1mes']==4, 'indrel_1mes', '4.0')

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


# concatenate two train dfs
train_fea_df2 = pd.concat([train_fea_df, vali_fea_df])
train_fea_df = train_fea_df2.reset_index(drop=True)
train_out_df2 = pd.concat([train_out_df, vali_out_df])
train_out_df = train_out_df2.reset_index(drop=True)

# # # ********************************************* # # #
# # # prepare the data from xgb trees
# # # ********************************************* # # #
# delete the country as it generates too many hot vectors
# organize the data so that each feature row contains all info and 6 months of product
# data. + 1 is for the total number of products column
train_prod_ary = np.zeros([train_fea_df.shape[0]/6, class_num*6 + 6])
pred_prod_ary = np.zeros([pred_fea_df.shape[0]/6, class_num*6+6])
# re-sample the feature data frame at every 6 rows
train_fea_df6 = train_fea_df.iloc[::6].reset_index(drop=True)
pred_fea_df6 = pred_fea_df.iloc[::6].reset_index(drop=True)
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
pred_num_ary = pred_fea_df6[train_numerical_list].values
total_df = pd.concat([train_fea_df6[train_dummy_list],
                      pred_fea_df6[train_dummy_list]])
dummy_df = pd.get_dummies(total_df[train_dummy_list])
train_dum_df = dummy_df[0:train_fea_df6.shape[0]]
pred_dum_df = dummy_df[train_fea_df6.shape[0]:]

train_dum_ary = train_dum_df.values
# if some value never appear in the training set, then delete it from the test set
# pred_dum_df_selected = pred_dum_df[train_dum_df.columns]
pred_dum_ary = pred_dum_df.values

# # # ********************************************* # # #
# # # set the tree parameters
# # # ********************************************* # # #
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.11
param['max_depth'] = 9
param['silent'] = 1
# param['nthread'] = 4
param['num_class'] = 4
param['colsample_bytree'] = 0.8
param['subsample'] = 0.8
# # # ********************************************* # # #
# # # get the hyper-classes labels
# # # ********************************************* # # #
num_tree = 1
num_round = 2000
p_pred = np.zeros((pred_fea_df.shape[0]/6,16))
for iter_tree in tqdm(range(0, num_tree)):
    hyper_class = genRandClass()
    # compute the hyper-classed product list and reshape it
    train_classed_ary = np.dot(train_fea_df.ix[:, info_num:], hyper_class.transpose())
    pred_classed_ary = np.dot(pred_fea_df.ix[:, info_num:], hyper_class.transpose())
    for i in range(0, train_fea_df.shape[0]/6):
        train_prod_ary[i, :-1] = train_classed_ary[6*i: 6*i+6, :].\
            reshape(class_num * 6)
        train_prod_ary[i, -6] = np.sum(train_fea_df.ix[6*i, info_num:])
        train_prod_ary[i, -5] = np.sum(train_fea_df.ix[6*i+1, info_num:])
        train_prod_ary[i, -4] = np.sum(train_fea_df.ix[6*i+2, info_num:])
        train_prod_ary[i, -3] = np.sum(train_fea_df.ix[6*i+3, info_num:])
        train_prod_ary[i, -2] = np.sum(train_fea_df.ix[6*i+4, info_num:])
        train_prod_ary[i, -1] = np.sum(train_fea_df.ix[6*i+5, info_num:])
    for i in range(0, pred_fea_df.shape[0]/6):
        pred_prod_ary[i, :-1] = pred_classed_ary[6*i: 6*i+6, :].\
            reshape(class_num * 6)
        pred_prod_ary[i, -1] = np.sum(pred_fea_df.ix[i, info_num:])
        pred_prod_ary[i, -6] = np.sum(pred_prod_ary.ix[6*i, info_num:])
        pred_prod_ary[i, -5] = np.sum(pred_prod_ary.ix[6*i+1, info_num:])
        pred_prod_ary[i, -4] = np.sum(pred_prod_ary.ix[6*i+2, info_num:])
        pred_prod_ary[i, -3] = np.sum(pred_prod_ary.ix[6*i+3, info_num:])
        pred_prod_ary[i, -2] = np.sum(pred_prod_ary.ix[6*i+4, info_num:])
        pred_prod_ary[i, -1] = np.sum(pred_prod_ary.ix[6*i+5, info_num:])

    # concatenate all features(products, info) into one array
    train_ary = np.concatenate((train_prod_ary, train_num_ary, train_dum_ary), axis=1)
    pred_ary = np.concatenate((pred_prod_ary, pred_num_ary, pred_dum_ary), axis=1)

    train_labels = genClassLabel(train_out_df, hyper_class)
    # train a tree
    xg_train = xgb.DMatrix(train_ary, label=train_labels)
    watchlist = [ (xg_train,'train')]

    bst = xgb.train(param, xg_train, num_round, watchlist );
    # get prediction
    xg_test = xgb.DMatrix(pred_ary)
    hc_pred = bst.predict( xg_test )
    # compute the real products added according to the prediction
    p_pred += np.dot(hc_pred, hyper_class)

np.savetxt("../input/pred_1205.csv", p_pred, delimiter=",")


product_list = pred_fea_df.columns[info_num:]
# seven_positions = np.zeros([pred_fea_df.shape[0]/6, 7])
final_prediction=[]
for i in range(0, pred_fea_df.shape[0]/6):
    seven_positions = np.argpartition(np.array(p_pred[i]), -7)[-7:]
    added_products_row = train_fea_df.columns[info_num:]
    added_products_list = [product_list[i] for i in seven_positions]
    final_prediction.append(" ".join(added_products_list))
    # final_prediction[i, seven_positions] = 1
# construct a dataframe
# fp_df = pd.Series({'ncodpers': pred_fea_df6.ncodpers.values.transpose,
#                    'added_products': final_prediction})

df_p = pd.DataFrame(final_prediction, columns=['added_products'])
df_i = pd.DataFrame(pred_fea_df6.ncodpers.values, columns=['ncodpers'])
df_pred = pd.concat([df_i, df_p], axis=1)
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
# full_score = 0.0
# for i in range(0, vali_out_df.shape[0]):
#     real_array = vali_out_df.ix[i].values
#     s = np.dot(real_array, full_prediction)
#     if s != 0:
#         full_score += s/sum(vali_out_df.ix[i])
# score /= total_customers
# print full_score
# # # ********************************************* # # #
# # # delete those customers that does not exist in test
# # # ********************************************* # # #
test_idx_df = pd.read_sql("select ncodpers from santander_test order by ncodpers", santanderCon)
# pred_idx_df = pred_fea_df6.reset_index(drop=True)
merged_df = pd.merge(test_idx_df, df_pred, how='left', on='ncodpers')
high_frequency_products = "ind_cco_fin_ult1 ind_cno_fin_ult1 ind_ecue_fin_ult1 " \
                          "ind_ecue_fin_ult1 ind_nomina_ult1 ind_nom_pens_ult1 " \
                          "ind_recibo_ult1"
null_list = np.where(merged_df.added_products.isnull().values==1)[0]
merged_df.ix[null_list[0], 'added_products'] = high_frequency_products
merged_df.to_csv('../output/sub_161206_2.csv', index=False)

