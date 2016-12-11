"""
Code based on BreakfastPirate Forum post
__author__ : SRK
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
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

mapping_dict = {
    'ind_empleado': {-99: 0, 'N': 1, 'B': 2, 'F': 3, 'A': 4, 'S': 5},
    'sexo': {'V': 0, 'H': 1, -99: 2},
    'ind_nuevo': {'0': 0, '1': 1, -99: 2},
    'indrel': {'1': 0, '99': 1, -99: 2},
    'indrel_1mes': {-99: 0, '1.0': 1, '1': 1, '2.0': 2, '2': 2, '3.0': 3, '3': 3, '4.0': 4, '4': 4, 'P': 5},
    'tiprel_1mes': {-99: 0, 'I': 1, 'A': 2, 'P': 3, 'R': 4, 'N': 5},
    'indresi': {-99: 0, 'S': 1, 'N': 2},
    'indext': {-99: 0, 'S': 1, 'N': 2},
    'conyuemp': {-99: 0, 'S': 1, 'N': 2},
    'indfall': {-99: 0, 'S': 1, 'N': 2},
    'tipodom': {-99: 0, '1': 1},
    'ind_actividad_cliente': {'0': 0, '1': 1, -99: 2},
    'segmento': {'02 - PARTICULARES': 0, '03 - UNIVERSITARIO': 1, '01 - TOP': 2, -99: 2},
    'pais_residencia': {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17,
                        'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73,
                        'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67,
                        'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20,
                        'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90,
                        'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118,
                        'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7,
                        'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2,
                        'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53,
                        'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24,
                        'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29,
                        'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14,
                        'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58,
                        'MZ': 27},
    'canal_entrada': {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12,
                      'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57,
                      'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35,
                      'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37,
                      'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133,
                      'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17,
                      'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121,
                      'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105,
                      'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65,
                      'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129,
                      'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60,
                      'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117,
                      'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132,
                      'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155,
                      'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152,
                      'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87,
                      'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38,
                      'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42,
                      'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83,
                      'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11}
}
cat_cols = list(mapping_dict.keys())

target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
               'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
target_cols = target_cols[2:]


def getTarget(row):
    tlist = []
    for col in target_cols:
        if row[col].strip() in ['', 'NA']:
            target = 0
        else:
            target = int(float(row[col]))
        tlist.append(target)
    return tlist


def getIndex(row, col):
    val = row[col].strip()
    if val not in ['', 'NA']:
        ind = mapping_dict[col][val]
    else:
        ind = mapping_dict[col][-99]
    return ind


def getAge(row):
    mean_age = 40.
    min_age = 20.
    max_age = 90.
    range_age = max_age - min_age
    age = row['age'].strip()
    if age == 'NA' or age == '':
        age = mean_age
    else:
        age = float(age)
        if age < min_age:
            age = min_age
        elif age > max_age:
            age = max_age
    return round((age - min_age) / range_age, 4)


def getCustSeniority(row):
    min_value = 0.
    max_value = 256.
    range_value = max_value - min_value
    missing_value = 0.
    cust_seniority = row['antiguedad'].strip()
    if cust_seniority == 'NA' or cust_seniority == '':
        cust_seniority = missing_value
    else:
        cust_seniority = float(cust_seniority)
        if cust_seniority < min_value:
            cust_seniority = min_value
        elif cust_seniority > max_value:
            cust_seniority = max_value
    return round((cust_seniority - min_value) / range_value, 4)


def getRent(row):
    min_value = 0.
    max_value = 1500000.
    range_value = max_value - min_value
    missing_value = 101850.
    rent = row['renta'].strip()
    if rent == 'NA' or rent == '':
        rent = missing_value
    else:
        rent = float(rent)
        if rent < min_value:
            rent = min_value
        elif rent > max_value:
            rent = max_value
    return round((rent - min_value) / range_value, 6)


def processData(in_file_name, cust_dict):
    x_vars_list = []
    y_vars_list = []
    for row in csv.DictReader(in_file_name):
        # use only the four months as specified by breakfastpirate #
        if row['fecha_dato'] not in ['2015-05-28', '2015-06-28', '2016-05-28', '2016-06-28']:
            continue
        # if the date is 05-28, only record its product use info, which will be used later
        # to calculate the added products at 0628
        cust_id = int(row['ncodpers'])
        if row['fecha_dato'] in ['2015-05-28', '2016-05-28']:
            target_list = getTarget(row)
            cust_dict[cust_id] = target_list[:]
            continue
        # if date is 06-28, get user attributes: index, age, seniority and rent
        x_vars = []
        for col in cat_cols:
            x_vars.append(getIndex(row, col))
        x_vars.append(getAge(row))
        x_vars.append(getCustSeniority(row))
        x_vars.append(getRent(row))
        
        if row['fecha_dato'] == '2016-06-28':
            prev_target_list = cust_dict.get(cust_id, [0] * 22)
            x_vars_list.append(x_vars + prev_target_list)
        elif row['fecha_dato'] == '2015-06-28':
            prev_target_list = cust_dict.get(cust_id, [0] * 22)
            target_list = getTarget(row)
            new_products = [max(x1 - x2, 0) for (x1, x2) in zip(target_list, prev_target_list)]
            if sum(new_products) > 0:
                for ind, prod in enumerate(new_products):
                    if prod > 0:
                        assert len(prev_target_list) == 22
                        x_vars_list.append(x_vars + prev_target_list)
                        y_vars_list.append(ind)

    return x_vars_list, y_vars_list, cust_dict




def runXGB(train_X, train_y, seed_val=0):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.035
    param['max_depth'] = 9
    param['silent'] = 1
    param['num_class'] = 22
    param['eval_metric'] = "mlogloss"
    # param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = 2000

    plst = list(param.items())
    xg_train = xgb.DMatrix(train_X, label=train_y)
    watchlist = [ (xg_train,'train')]
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
    # del input_df['ind_cder_fin_ult1']
    del input_df['ind_deco_fin_ult1']
    del input_df['ind_deme_fin_ult1']
    # del input_df['ind_hip_fin_ult1']
    # del input_df['ind_pres_fin_ult1']
    # del input_df['ind_viv_fin_ult1']

def convertNum(intput_df):
    intput_df['age'] = pd.to_numeric(intput_df['age'], errors='coerce')
    intput_df['antiguedad'] = pd.to_numeric(intput_df['antiguedad'], errors='coerce')

def modelfit(alg, train_X, train_y,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_X, label=train_y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(train_X, train_y, eval_metric='mlogloss')

    #Predict training set:
    dtrain_predictions = alg.predict(train_X)
    # dtrain_predprob = alg.predict_proba(train_X)[:,1]

    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(train_y, dtrain_predictions)
    # print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)


if __name__ == "__main__":
    product_num = 20
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
    # compute the total number of products for each month
    total_train_prod = np.sum(train_fea_df.ix[:, info_num:].values, axis=1)
    total_pred_prod = np.sum(pred_fea_df.ix[:, info_num:].values, axis=1)
    train_fea_list = []
    train_out_list = []
    train_prod_ary = np.zeros([train_out_df.shape[0], product_num*2+2])
    for i in tqdm(range(0, train_out_df.shape[0])):
        train_prod_ary[i,:-2] = train_fea_df.ix[6 * i: 6 * i + 1, info_num:].\
            values.reshape(product_num*2)
        train_prod_ary[i, -2] = total_train_prod[6*i]
        train_prod_ary[i, -1] = total_train_prod[6*i+1]
        train_fea_element = np.concatenate(
            (train_prod_ary[i], train_num_ary[i], train_dum_ary[i]))
        # one_p = np.where(train_out_df.ix[i].values==1)
        # this_element = train_fea_element * one_p.shape[0]

        for j in range(0, product_num):

            if train_out_df.ix[i, j] == 1:
                train_fea_list.append(train_fea_element)
                train_out_list.append(j)
    # print("Building model..")
    train_X = np.array(train_fea_list)
    train_y = np.array(train_out_list)
    xgb1 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=4000,
        max_depth=8,
        min_child_weight=1,
        gamma=0,
        subsample=0.7,
        colsample_bytree=0.7,
        objective='multi:softprob',
        # nthread=4,
        scale_pos_weight=1,
        seed=0)
    early_stopping = 50
    xgb_param = xgb1.get_xgb_params()
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5,
                  metrics='mlogloss', early_stopping_rounds=early_stopping)
    xgb1.set_params(n_estimators=cvresult.shape[0])
    modelfit(xgb1, train_X, train_y)
    # generate the prediction set with non-repeated attibutes
    # pred_fea_list = []
    # pred_prod_ary = np.zeros([pred_fea_df.shape[0], product_num*2+2])
    # for i in tqdm(range(0, pred_fea_df6.shape[0])):
    #     pred_prod_ary[i,:-2] = pred_fea_df.ix[6 * i: 6 * i + 1, info_num:].\
    #         values.reshape(product_num*2)
    #     pred_prod_ary[i, -2] = total_pred_prod[6*i]
    #     pred_prod_ary[i, -1] = total_pred_prod[6*i+1]
    #     pred_fea_element = np.concatenate(
    #         (pred_prod_ary[i], pred_num_ary[i], pred_dum_ary[i]))
    #     # one_p = np.where(train_out_df.ix[i].values==1)
    #     # this_element = train_fea_element * one_p.shape[0]
    #     pred_fea_list.append(pred_fea_element)
    # # del train_X, train_y
    # pred_X = np.array(pred_fea_list)
    # print("Predicting..")
    # xgtest = xgb.DMatrix(pred_X)
    # # preds = model.predict(xgtest)
    # preds = preds[:, 0:16]
    # del test_X, xgtest
    # print(datetime.datetime.now() - start_time)

    # print("Getting the top products..")
    # target_cols = np.array(target_cols)
    # target_cols = np.array(train_fea_df.columns[16:])
    # preds_s = np.argsort(preds, axis=1)
    # preds_s = np.fliplr(preds_s)[:, :7]
    # test_id = np.array(pd.read_csv("../input/test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
    # final_preds = [" ".join(list(target_cols[pred])) for pred in preds_s]
    # predicted_df = pd.DataFrame({'ncodpers': pred_fea_df6.ncodpers.values
    #                              , 'added_products': final_preds})
    # # out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    # # out_df.to_csv('sub_xgb_new.csv', index=False)
    # # print(datetime.datetime.now() - start_time)
    # test_idx_df = pd.read_sql("select ncodpers from santander_test order by ncodpers", santanderCon)
    # # pred_idx_df = pred_fea_df6.reset_index(drop=True)
    # merged_df = pd.merge(test_idx_df, predicted_df, how='left', on='ncodpers')
    # high_frequency_products = "ind_cco_fin_ult1 ind_cno_fin_ult1 ind_ecue_fin_ult1 " \
    #                           "ind_ecue_fin_ult1 ind_nomina_ult1 ind_nom_pens_ult1 " \
    #                           "ind_recibo_ult1"
    # null_list = np.where(merged_df.added_products.isnull().values==1)[0]
    # merged_df.ix[null_list[0], 'added_products'] = high_frequency_products
    # merged_df.to_csv('../output/sub_161209.csv', index=False)
