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
    # del input_df['ind_cder_fin_ult1']
    del input_df['ind_deco_fin_ult1']
    del input_df['ind_deme_fin_ult1']
    # del input_df['ind_hip_fin_ult1']
    # del input_df['ind_pres_fin_ult1']
    # del input_df['ind_viv_fin_ult1']

def convertNum(intput_df):
    intput_df['age'] = pd.to_numeric(intput_df['age'], errors='coerce')
    intput_df['antiguedad'] = pd.to_numeric(intput_df['antiguedad'], errors='coerce')

product_num = 20
info_num = 16
class_num = 4
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


# concatenate two train dfs
train_fea_df2 = pd.concat([train_fea_df, vali_fea_df])
train_fea_df = train_fea_df2.reset_index(drop=True)
train_out_df2 = pd.concat([train_out_df, vali_out_df])
train_out_df = train_out_df2.reset_index(drop=True)
p_pred = np.loadtxt('../input/pred_1213.csv', delimiter=",")
target_cols = np.array(train_fea_df.columns[16:])
preds_s = np.argsort(p_pred, axis=1)
preds_s = np.fliplr(preds_s)[:, :7]

for i in range(0, product_num):
    product_name = train_fea_df.columns[info_num + i]
    occurs = np.where(vali_out_df.ix[:, i] == 1)[0]
