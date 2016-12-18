"""
"""
import csv
import datetime
from operator import sub
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, ensemble
import sqlite3 as sql
from tqdm import tqdm

# a dictionary that maps country to its development condition
# home is home country, wd means well-developed, ld:less-developed, di:developing
dict_country = {'ES': 'home', 'FR': 'wd', 'IT': 'wd', 'GB': 'wd', 'DE': 'wd', 'CR': 'di',
                'PT': 'ld', 'US': 'wd', 'CH': 'ld', 'MX': 'ld', 'MA': 'di', 'CL': 'di',
                'NL': 'wd', 'KE': 'di', 'BR': 'ld', 'AR': 'ld', 'AT': 'wd', 'IN': 'di',
                'BE': 'wd', 'AU': 'wd', 'PA': 'ld', 'VE': 'di', 'BO': 'di', 'GR': 'ld',
                'ZA': 'ld', 'RU': 'ld', 'JP': 'wd', 'CN': 'ld', 'SG': 'wd', 'NG': 'di',
                'NZ': 'wd', 'PE': 'di', 'DK': 'wd', 'NO': 'wd', 'AE': 'ld', 'TR': 'ld',
                'GA': 'di', 'CO': 'ld', 'CA': 'wd', 'AD': 'wd', 'BY': 'ld', 'HK': 'wd',
                'IE': 'wd', 'SE': 'wd', 'PL': 'wd', 'VN': 'ld', 'FI': 'wd', 'PR': 'ld',
                'LU': 'wd', 'AO': 'di', 'HN': 'di', 'QA': 'ld', 'OM': 'ld', 'CG': 'di',
                'CZ': 'ld', 'CM': 'di', 'ET': 'di', 'SA': 'ld', 'CI': 'di', 'MT': 'wd',
                'IL': 'wd', 'GQ': 'di', 'RO': 'ld'}

country_reduce_map = lambda x: dict_country[x]

dict_date = {'2016-05-28': 'May', '2016-02-28': 'Feb', '2016-01-28': 'Jan',
             '2015-12-28': 'Dec', '2015-11-28': 'Nov', '2015-10-28': 'OCT',
             '2015-09-28': 'Sep', '2015-08-28': 'Aug', '2015-07-28': 'Jul',
             '2015-06-28': 'Jun', '2015-05-28': 'May', '2015-04-28': 'Apr',
             '2015-03-28': 'Mar', '2015-02-28': 'Feb', '2015-01-28': 'Jan',
             '2016-03-28': 'Mar', '2016-06-28': 'Jun', '2016-04-28': 'Apr'}

date_reduce_map = lambda x: dict_date[x]

def runXGB(train_X, train_y, train_weight, r, seed_val=0):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05
    param['max_depth'] = 9
    param['silent'] = 1
    param['num_class'] = 20
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    param['gamma'] = 2
    num_rounds = r

    plst = list(param.items())
    xg_train = xgb.DMatrix(train_X, label=train_y, weight=train_weight)
    watchlist = [(xg_train, 'train')]
    model = xgb.train(plst, xg_train, num_rounds, watchlist)
    return model


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


if __name__ == "__main__":

    product_num = 20
    info_num = 19
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

    train_fea_df['country'] = train_fea_df['pais_residencia'].map(country_reduce_map)
    vali_fea_df['country'] = vali_fea_df['pais_residencia'].map(country_reduce_map)
    train_fea_df['date'] = train_fea_df['fecha_dato'].map(date_reduce_map)
    vali_fea_df['date'] = vali_fea_df['fecha_dato'].map(date_reduce_map)

    convertNum(train_fea_df)
    convertNum(vali_fea_df)
    # re organize the columns
    current_cols = train_fea_df.columns.tolist()
    target_cols = current_cols[-2:] + current_cols[:-2]
    train_fea_df = train_fea_df[target_cols]
    vali_fea_df = vali_fea_df[target_cols]
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
    train_dummy_list = ['date', 'ind_empleado',
                        'sexo', 'indrel_1mes', 'tiprel_1mes',
                        'indresi', 'conyuemp', 'segmento', 'country']
    train_numerical_list = ['age', 'antiguedad', 'indrel', 'ind_actividad_cliente',
                            'renta', 'cod_prov', 'ind_nuevo']
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
    train_weight_i = 1.0/np.sum(train_out_df, axis=1)
    # vali_weight_i = 1.0/np.sum(vali_out_df, axis=1)
    train_fea_list = []
    train_out_list = []
    train_weight_list = []
    train_prod_ary = np.zeros([train_out_df.shape[0], product_num * 2 + 2])
    for i in tqdm(range(0, train_out_df.shape[0])):
        train_prod_ary[i, :-2] = train_fea_df.ix[6 * i: 6 * i + 1, info_num:]. \
            values.reshape(product_num * 2)
        train_prod_ary[i, -2] = total_train_prod[6 * i]
        train_prod_ary[i, -1] = total_train_prod[6 * i + 1]
        train_fea_element = np.concatenate(
            (train_prod_ary[i], train_num_ary[i], train_dum_ary[i]))
        # one_p = np.where(train_out_df.ix[i].values==1)
        # this_element = train_fea_element * one_p.shape[0]

        for j in range(0, product_num):

            if train_out_df.ix[i, j] == 1:
                train_fea_list.append(train_fea_element)
                train_out_list.append(j)
                train_weight_list.append(train_weight_i[i])
    # print("Building model..")
    train_X = np.array(train_fea_list)
    train_y = np.array(train_out_list)
    train_weight = np.array(train_weight_list)
    # generate the prediction set with non-repeated attributes
    vali_fea_list = []
    vali_prod_ary = np.zeros([vali_fea_df.shape[0], product_num * 2 + 2])
    for i in tqdm(range(0, vali_fea_df6.shape[0])):
        vali_prod_ary[i, :-2] = vali_fea_df.ix[6 * i: 6 * i + 1, info_num:]. \
            values.reshape(product_num * 2)
        vali_prod_ary[i, -2] = total_vali_prod[6 * i]
        vali_prod_ary[i, -1] = total_vali_prod[6 * i + 1]
        vali_fea_element = np.concatenate(
            (vali_prod_ary[i], vali_num_ary[i], vali_dum_ary[i]))
        # one_p = np.where(train_out_df.ix[i].values==1)
        # this_element = train_fea_element * one_p.shape[0]
        vali_fea_list.append(vali_fea_element)
    vali_X = np.array(vali_fea_list)
    xgtest = xgb.DMatrix(vali_X)

    model = runXGB(train_X, train_y, train_weight, 220, seed_val=0)
    preds = model.predict(xgtest)

    final_prediction = np.zeros(vali_out_df.shape)
    for i in range(0, vali_out_df.shape[0]):
        seven_positions = np.argpartition(np.array(preds[i]), -7)[-7:]
        final_prediction[i, seven_positions] = 1
    # np.savetxt("../input/pred_1130.csv", final_prediction, delimiter=",")
    total_customers = 897377
    score = 0.0
    for i in range(0, vali_out_df.shape[0]):
        real_array = vali_out_df.ix[i].values
        s = np.dot(real_array, final_prediction[i])
        if s != 0:
            score += s/sum(vali_out_df.ix[i])
    score /= total_customers
    print score
