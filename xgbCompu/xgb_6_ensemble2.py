# in this script, the hyper class idea is totally abandoned
# binary prediction is performed over 16 products and 7 with
# highest scores are picked out
import xgboost as xgb
import pandas as pd
import sqlite3 as sql
from tqdm import tqdm
import numpy as np
import random
from collections import defaultdict

dict_country = {'ES': 'home', 'FR': 'wd', 'IT': 'wd', 'GB': 'wd', 'DE': 'wd', 'CR': 'di','PT': 'ld', 'US': 'wd', 'CH': 'ld', 'MX': 'ld', 'MA': 'di', 'CL': 'di','NL': 'wd', 'KE': 'di', 'BR': 'ld', 'AR': 'ld', 'AT': 'wd', 'IN': 'di','BE': 'wd', 'AU': 'wd', 'PA': 'ld', 'VE': 'di', 'BO': 'di', 'GR': 'ld','ZA': 'ld', 'RU': 'ld', 'JP': 'wd', 'CN': 'ld', 'SG': 'wd', 'NG': 'di','NZ': 'wd', 'PE': 'di', 'DK': 'wd', 'NO': 'wd', 'AE': 'ld', 'TR': 'ld','GA': 'di', 'CO': 'ld', 'CA': 'wd', 'AD': 'wd', 'BY': 'ld', 'HK': 'wd','IE': 'wd',
                'SE': 'wd', 'PL': 'wd', 'VN': 'ld', 'FI': 'wd', 'PR': 'ld','LU': 'wd', 'AO': 'di', 'HN': 'di', 'QA': 'ld', 'OM': 'ld', 'CG': 'di','CZ': 'ld', 'CM': 'di', 'ET': 'di', 'SA': 'ld', 'CI': 'di', 'MT': 'wd','IL': 'wd', 'GQ': 'di', 'RO': 'ld', 'AL': 'ld', 'BA': 'ld', 'BG': 'ld', 'BM': 'ld', 'BZ': 'ld', 'CD':'di',
                'CF': 'di','CU': 'di','DJ': 'di','DO': 'di','DZ': 'ld','EC': 'ld','EE': 'ld','EG': 'ld','GE': 'ld','GH': 'di','GI': 'ld','GM': 'di','GN': 'di','GT': 'ld','GW': 'di','HR': 'ld','HU': 'ld','IS': 'wd','JM': 'ld','KH': 'ld','KR': 'wd','KW': 'ld','KZ': 'ld','LB': 'ld','LT': 'ld','LV': 'ld','LY': 'di','MD': 'ld','MK': 'ld','ML': 'di','MM': 'ld','MR': 'di','MZ': 'di','NI': 'ld','PH': 'ld','PK': 'di','PY': 'ld','RS': 'ld','SK': 'ld','SL': 'di','SN': 'di','SV': 'di','TG': 'di','TH': 'ld','TN': 'ld','TW': 'wd','UA': 'ld','UY': 'ld','ZW': 'di'}

country_reduce_map = lambda x: dict_country[x]

dict_date = {'2016-05-28': 'May', '2016-02-28': 'Feb', '2016-01-28': 'Jan','2015-12-28': 'Dec', '2015-11-28': 'Nov', '2015-10-28': 'OCT','2015-09-28': 'Sep', '2015-08-28': 'Aug', '2015-07-28': 'Jul','2015-06-28': 'Jun', '2015-05-28': 'May', '2015-04-28': 'Apr','2015-03-28': 'Mar', '2015-02-28': 'Feb', '2015-01-28': 'Jan','2016-03-28': 'Mar', '2016-06-28': 'Jun', '2016-04-28': 'Apr'}

date_reduce_map = lambda x: dict_date[x]

dict_prov_off = defaultdict(lambda: 1084) #"CORU\xd1A, A":  1084 ,
dict_prov_off["ALAVA"]=  341 ;dict_prov_off["ALBACETE"]=  402 ;dict_prov_off["ALICANTE"]=  1523 ;dict_prov_off["ALMERIA"]=  625 ;dict_prov_off["ASTURIAS"]=  1045 ;dict_prov_off["AVILA"]=  184 ;dict_prov_off["BADAJOZ"]=  840 ;dict_prov_off["BALEARS, ILLES"]=  1372 ;dict_prov_off["BARCELONA"]=  5474 ;dict_prov_off["BURGOS"]=  470 ;dict_prov_off["CACERES"]=  556 ;dict_prov_off["CADIZ"]=  707 ;dict_prov_off["CANTABRIA"]=  567 ;dict_prov_off["CASTELLON"]=  548 ;dict_prov_off["CEUTA"]=  30 ;dict_prov_off["CIUDAD REAL"]=  510 ;dict_prov_off["CORDOBA"]=  791 ;dict_prov_off["CORU\xd1A, A"]=  1084 ;dict_prov_off["CUENCA"]=  307 ;dict_prov_off["GIRONA"]=  804 ;dict_prov_off["GRANADA"]=  232 ;dict_prov_off["GUADALAJARA"]=  258 ;dict_prov_off["GIPUZKOA"]=  800 ;dict_prov_off["HUELVA"]=  395 ;dict_prov_off["HUESCA"]=  342 ;dict_prov_off["JAEN"]=  674 ;dict_prov_off["RIOJA, LA"]=  426 ;dict_prov_off["PALMAS, LAS"]=  705 ;
dict_prov_off["LEON"]=  570 ;dict_prov_off["LERIDA"]=  635 ;dict_prov_off["LUGO"]=  345 ;dict_prov_off["MADRID"]=  6531 ;dict_prov_off["MALAGA"]=  1256 ;dict_prov_off["MELILLA"]=  24 ;dict_prov_off["MURCIA"]=  1200 ;dict_prov_off["NAVARRA"]=  699 ;dict_prov_off["OURENSE"]=  297 ;dict_prov_off["PALENCIA"]=  217 ;dict_prov_off["PONTEVEDRA"]=  685 ;dict_prov_off["SALAMANCA"]=  450 ;dict_prov_off["SEGOVIA"]=  186 ;dict_prov_off["SEVILLA"]=  300 ;dict_prov_off["SORIA"]=  163 ;dict_prov_off["TARRAGONA"]=  848 ;dict_prov_off["SANTA CRUZ DE TENERIFE"]=  701 ;dict_prov_off["TERUEL"]=  253 ;dict_prov_off["TOLEDO"]=  719 ;dict_prov_off["VALENCIA"]=  2261 ;dict_prov_off["VALLADOLID"]=  574 ;dict_prov_off["BIZKAIA"]=  1150 ;dict_prov_off["ZAMORA"]=  246 ;dict_prov_off["ZARAGOZA"]=  1191 ;

prov_off_map = lambda x: dict_prov_off[x]

dict_prov_opa = defaultdict(lambda: 0.136352201)
dict_prov_opa["ALAVA"]=  0.112281857 ;dict_prov_opa["ALBACETE"]=  0.026932869 ;dict_prov_opa["ALICANTE"]=  0.261818807 ;dict_prov_opa["ALMERIA"]=  0.071225071 ;dict_prov_opa["ASTURIAS"]=  0.098547718 ;dict_prov_opa["AVILA"]=  0.022857143 ;dict_prov_opa["BADAJOZ"]=  0.0385923 ;dict_prov_opa["BALEARS, ILLES"]=  0.274839744 ;dict_prov_opa["BARCELONA"]=  0.708333333 ;dict_prov_opa["BURGOS"]=  0.03288553 ;dict_prov_opa["CACERES"]=  0.027984699 ;dict_prov_opa["CADIZ"]=  0.095077999 ;dict_prov_opa["CANTABRIA"]=  0.106558917 ;dict_prov_opa["CASTELLON"]=  0.082629674 ;dict_prov_opa["CEUTA"]=  1.5 ;dict_prov_opa["CIUDAD REAL"]=  0.025740675 ;dict_prov_opa["CORDOBA"]=  0.057439547 ;dict_prov_opa["CORU\xd1A, A"]=  0.136352201 ;dict_prov_opa["CUENCA"]=  0.017911319 ;dict_prov_opa["GIRONA"]=  0.421162913 ; dict_prov_opa["GRANADA"]=  0.018344271 ;
dict_prov_opa["GUADALAJARA"]=  0.021123301 ;dict_prov_opa["GIPUZKOA"]=  0.078988942 ;dict_prov_opa["HUELVA"]=  0.025262215 ;dict_prov_opa["HUESCA"]=  0.021872602 ;dict_prov_opa["JAEN"]=  0.049940723 ;dict_prov_opa["RIOJA, LA"]=  0.08444004 ;dict_prov_opa["PALMAS, LAS"]=  0.17338908 ;dict_prov_opa["LEON"]=  0.036583018 ;dict_prov_opa["LERIDA"]=  0.052164627 ;dict_prov_opa["LUGO"]=  0.035004058 ;dict_prov_opa["MADRID"]=  0.813527653 ;dict_prov_opa["MALAGA"]=  0.171866448 ;dict_prov_opa["MELILLA"]=  0.857142857 ;dict_prov_opa["MURCIA"]=  0.10607266 ;dict_prov_opa["NAVARRA"]=  0.067269753 ;dict_prov_opa["OURENSE"]=  0.040835969 ;dict_prov_opa["PALENCIA"]=  0.026949826 ;dict_prov_opa["PONTEVEDRA"]=  0.152391546 ;dict_prov_opa["SALAMANCA"]=  0.036437247 ;dict_prov_opa["SEGOVIA"]=  0.026874729 ;dict_prov_opa["SEVILLA"]=  0.021373611 ;
dict_prov_opa["SORIA"]=  0.015816029 ;dict_prov_opa["TARRAGONA"]=  0.134539108 ;dict_prov_opa["SANTA CRUZ DE TENERIFE"]=  0.207335108 ;dict_prov_opa["TERUEL"]=  0.017083052 ;dict_prov_opa["TOLEDO"]=  0.04677944 ;dict_prov_opa["VALENCIA"]=  0.20923561 ;dict_prov_opa["VALLADOLID"]=  0.053118638 ;dict_prov_opa["BIZKAIA"]=  0.141800247 ;dict_prov_opa["ZAMORA"]=  0.023293249 ;dict_prov_opa["ZARAGOZA"]=  0.068947551 ;

prov_opa_map = lambda x: dict_prov_opa[x]

dict_prov_gdp = defaultdict(lambda: 21898)
dict_prov_gdp["ALAVA"]=  35175 ;dict_prov_gdp["ALBACETE"]=  18113 ;dict_prov_gdp["ALICANTE"]=  17405 ;dict_prov_gdp["ALMERIA"]=  16855 ;dict_prov_gdp["ASTURIAS"]=  21310 ;dict_prov_gdp["AVILA"]=  19011 ;dict_prov_gdp["BADAJOZ"]=  15617 ;dict_prov_gdp["BALEARS, ILLES"]=  23769 ;dict_prov_gdp["BARCELONA"]=  26531 ;dict_prov_gdp["BURGOS"]=  27128 ;dict_prov_gdp["CACERES"]=  15715 ;dict_prov_gdp["CADIZ"]=  16916 ;dict_prov_gdp["CANTABRIA"]=  22055 ;dict_prov_gdp["CASTELLON"]=  22597 ;dict_prov_gdp["CEUTA"]=  19555 ;dict_prov_gdp["CIUDAD REAL"]=  18214 ;dict_prov_gdp["CORDOBA"]=  16396 ;dict_prov_gdp["CORU\xd1A, A"]=  21898 ;
dict_prov_gdp["CUENCA"]=  18549 ;dict_prov_gdp["GIRONA"]=  26722 ;dict_prov_gdp["GRANADA"]=  16133 ;dict_prov_gdp["GUADALAJARA"]=  19584 ;dict_prov_gdp["GIPUZKOA"]=  31442 ;dict_prov_gdp["HUELVA"]=  17959 ;dict_prov_gdp["HUESCA"]=  26258 ;dict_prov_gdp["JAEN"]=  15858 ;dict_prov_gdp["RIOJA, LA"]=  25537 ;dict_prov_gdp["PALMAS, LAS"]=  19438 ;dict_prov_gdp["LEON"]=  20688 ;dict_prov_gdp["LERIDA"]=  26943 ;dict_prov_gdp["LUGO"]=  19459 ;dict_prov_gdp["MADRID"]=  29576 ;dict_prov_gdp["MALAGA"]=  17267 ;dict_prov_gdp["MELILLA"]=  17824 ;dict_prov_gdp["MURCIA"]=  18470 ;dict_prov_gdp["NAVARRA"]=  29134 ;dict_prov_gdp["OURENSE"]=  19305 ;dict_prov_gdp["PALENCIA"]=  23019 ;dict_prov_gdp["PONTEVEDRA"]=  19548 ;
dict_prov_gdp["SALAMANCA"]=  19264 ;dict_prov_gdp["SEGOVIA"]=  21769 ;dict_prov_gdp["SEVILLA"]=  18223 ;dict_prov_gdp["SORIA"]=  23816 ;dict_prov_gdp["TARRAGONA"]=  26792 ;dict_prov_gdp["SANTA CRUZ DE TENERIFE"]=  19205 ;dict_prov_gdp["TERUEL"]=  24996 ;dict_prov_gdp["TOLEDO"]=  17450 ;dict_prov_gdp["VALENCIA"]=  21091 ;dict_prov_gdp["VALLADOLID"]=  24176 ;dict_prov_gdp["BIZKAIA"]=  28618 ;dict_prov_gdp["ZAMORA"]=  19132 ;dict_prov_gdp["ZARAGOZA"]=  25150 ;

prov_gdp_map = lambda x: dict_prov_gdp[x]

dict_prov_opp = defaultdict(lambda: 0.005225381)
dict_prov_opp["ALAVA"]=  0.00105923 ;dict_prov_opp["ALBACETE"]=  0.001012628 ;dict_prov_opp["ALICANTE"]=  0.000815205 ;dict_prov_opp["ALMERIA"]=  0.000890718 ;dict_prov_opp["ASTURIAS"]=  0.000984219 ;dict_prov_opp["AVILA"]=  0.001097728 ;dict_prov_opp["BADAJOZ"]=  0.001215761 ;dict_prov_opp["BALEARS, ILLES"]=  0.001243374 ;dict_prov_opp["BARCELONA"]=  0.000990987 ;dict_prov_opp["BURGOS"]=  0.001280689 ;dict_prov_opp["CACERES"]=  0.001360381 ;dict_prov_opp["CADIZ"]=  0.000570081 ;dict_prov_opp["CANTABRIA"]=  0.000963211 ;dict_prov_opp["CASTELLON"]=  0.000932753 ;dict_prov_opp["CEUTA"]=  0.000353082 ;dict_prov_opp["CIUDAD REAL"]=  0.0009815 ;dict_prov_opp["CORDOBA"]=  0.00098949 ;dict_prov_opp["CORU\xd1A, A"]=  0.005225381 ;
dict_prov_opp["CUENCA"]=  0.000271027 ;dict_prov_opp["GIRONA"]=  0.001063273 ;dict_prov_opp["GRANADA"]=  0.000252323 ;dict_prov_opp["GUADALAJARA"]=  0.001010038 ;dict_prov_opp["GIPUZKOA"]=  0.00111865 ;dict_prov_opp["HUELVA"]=  0.000760743 ;dict_prov_opp["HUESCA"]=  0.001520615 ;dict_prov_opp["JAEN"]=  0.001022711 ;dict_prov_opp["RIOJA, LA"]=  0.001335415 ;dict_prov_opp["PALMAS, LAS"]=  0.000640893 ;dict_prov_opp["LEON"]=  0.001175995 ;dict_prov_opp["LERIDA"]=  0.001449768 ;dict_prov_opp["LUGO"]=  0.001006659 ;dict_prov_opp["MADRID"]=  0.001011835 ;dict_prov_opp["MALAGA"]=  0.000773884 ;dict_prov_opp["MELILLA"]=  0.000283993 ;dict_prov_opp["MURCIA"]=  0.000818097 ;dict_prov_opp["NAVARRA"]=  0.001090841 ;
dict_prov_opp["OURENSE"]=  0.000919687 ;dict_prov_opp["PALENCIA"]=  0.001170202 ;dict_prov_opp["PONTEVEDRA"]=  0.000720358 ;dict_prov_opp["SALAMANCA"]=  0.001314026 ;dict_prov_opp["SEGOVIA"]=  0.001325117 ;dict_prov_opp["SEVILLA"]=  0.000154531 ;dict_prov_opp["SORIA"]=  0.001735483 ;dict_prov_opp["TARRAGONA"]=  0.001058727 ;dict_prov_opp["SANTA CRUZ DE TENERIFE"]=  0.000697664 ;dict_prov_opp["TERUEL"]=  0.00151478 ;dict_prov_opp["TOLEDO"]=  0.001028412 ;dict_prov_opp["VALENCIA"]=  0.000887052 ;dict_prov_opp["VALLADOLID"]=  0.001084744 ;dict_prov_opp["BIZKAIA"]=  0.000998346 ;dict_prov_opp["ZAMORA"]=  0.001544227 ;dict_prov_opp["ZARAGOZA"]=  0.00124048 ;

prov_opp_map = lambda x: dict_prov_opp[x]


def deleteFeatures(input_df):
    # delete unwanted columns
    del input_df['fecha_alta']
    del input_df['ult_fec_cli_1t']
    del input_df['canal_entrada']
    # del input_df['nomprov']
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
info_num = 24
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
# del pred_fea_df['pais_residencia']
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

train_fea_df['country'] = train_fea_df['pais_residencia'].map(country_reduce_map)
vali_fea_df['country'] = vali_fea_df['pais_residencia'].map(country_reduce_map)
train_fea_df['date'] = train_fea_df['fecha_dato'].map(date_reduce_map)
vali_fea_df['date'] = vali_fea_df['fecha_dato'].map(date_reduce_map)
pred_fea_df['country'] = pred_fea_df['pais_residencia'].map(country_reduce_map)
pred_fea_df['date'] = pred_fea_df['fecha_dato'].map(date_reduce_map)

# deal with province name
train_fea_df[train_fea_df['nomprov'].isnull()].nomprov = train_fea_df.ix[0, 'nomprov']
train_fea_df['office'] = train_fea_df['nomprov'].map(prov_off_map)
train_fea_df['opa'] = train_fea_df['nomprov'].map(prov_opa_map)
train_fea_df['opp'] = train_fea_df['nomprov'].map(prov_opp_map)
train_fea_df['gdp'] = train_fea_df['nomprov'].map(prov_gdp_map)
vali_fea_df[vali_fea_df['nomprov'].isnull()].nomprov = vali_fea_df.ix[0, 'nomprov']
vali_fea_df['office'] = vali_fea_df['nomprov'].map(prov_off_map)
vali_fea_df['opa'] = vali_fea_df['nomprov'].map(prov_opa_map)
vali_fea_df['opp'] = vali_fea_df['nomprov'].map(prov_opp_map)
vali_fea_df['gdp'] = vali_fea_df['nomprov'].map(prov_gdp_map)
pred_fea_df[pred_fea_df['nomprov'].isnull()].nomprov = pred_fea_df.ix[0, 'nomprov']
pred_fea_df['office'] = pred_fea_df['nomprov'].map(prov_off_map)
pred_fea_df['opa'] = pred_fea_df['nomprov'].map(prov_opa_map)
pred_fea_df['opp'] = pred_fea_df['nomprov'].map(prov_opp_map)
pred_fea_df['gdp'] = pred_fea_df['nomprov'].map(prov_gdp_map)


convertNum(train_fea_df)
convertNum(vali_fea_df)

# reorganize the columns
current_cols = train_fea_df.columns.tolist()
target_cols = current_cols[-6:] + current_cols[:-6]
train_fea_df = train_fea_df[target_cols]
vali_fea_df = vali_fea_df[target_cols]
pred_fea_df = pred_fea_df[target_cols]

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
train_dummy_list = ['date', 'ind_actividad_cliente',
                    'sexo', 'tiprel_1mes', 'indrel', 'conyuemp', 'segmento', 'country']
train_numerical_list = ['age', 'antiguedad',
                        'renta', 'office', 'opa', 'gdp', 'opp']
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
train_weight = 1.0/np.sum(train_out_df, axis=1)

p_pred = np.zeros((pred_fea_df.shape[0]/6,product_num))
for seed_num in range(0, 800, 20):
    # # # ********************************************* # # #
    # # # set the tree parameters
    # # # ********************************************* # # #
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'binary:logitraw'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 13
    param['silent'] = 1
    # param['nthread'] = 4
    # param['num_class'] = 4
    param['colsample_bytree'] = 0.8
    param['subsample'] = 0.8
    param['seed'] = seed_num
    param['gamma'] = 6

    num_round = 300
    train_prod_ary = np.zeros([train_fea_df.shape[0]/6, 12])
    pred_prod_ary = np.zeros([pred_fea_df.shape[0]/6, 12])
    for iter_products in tqdm(range(0, product_num)):
        # get the name of the product to be predicted
        product_name = train_fea_df.columns[info_num + iter_products]
        # get the 6-month history of that product and total # of products
        train_prod_info = np.stack((train_fea_df[product_name].values,
            total_train_prod)).transpose()
        pred_prod_info = np.stack((pred_fea_df[product_name].values,
            total_pred_prod)).transpose()
        for i in range(0, train_fea_df.shape[0]/6):
            train_prod_ary[i] = train_prod_info[6*i: 6*i+6, :].reshape(12)
        for i in range(0, pred_fea_df.shape[0]/6):
            pred_prod_ary[i] = pred_prod_info[6*i: 6*i+6, :].reshape(12)

        # concatenate all features(products, info) into one array
        train_ary = np.concatenate((train_prod_ary, train_num_ary, train_dum_ary), axis=1)
        pred_ary = np.concatenate((pred_prod_ary, pred_num_ary, pred_dum_ary), axis=1)

        train_labels = train_out_df[product_name].values
        # train a tree
        xg_train = xgb.DMatrix(train_ary, label=train_labels)
        watchlist = [ (xg_train,'train')]

        bst = xgb.train(param, xg_train, num_round, watchlist );
        # get prediction
        xg_test = xgb.DMatrix(pred_ary)
        hc_pred = bst.predict( xg_test )
        # compute the real products added according to the prediction
        p_pred[:, iter_products] += hc_pred

np.savetxt("../input/pred_1217_ens1.csv", p_pred, delimiter=",")


# kaggle submission generation
target_cols = np.array(train_fea_df.columns[info_num:])
preds_s = np.argsort(p_pred, axis=1)
preds_s = np.fliplr(preds_s)[:, :7]
final_preds = [" ".join(list(target_cols[pred])) for pred in preds_s]
predicted_df = pd.DataFrame({'ncodpers': pred_fea_df6.ncodpers.values
                             , 'added_products': final_preds})
# out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
# out_df.to_csv('sub_xgb_new.csv', index=False)
# print(datetime.datetime.now() - start_time)
test_idx_df = pd.read_sql("select ncodpers from santander_test order by ncodpers", santanderCon)
# pred_idx_df = pred_fea_df6.reset_index(drop=True)
merged_df = pd.merge(test_idx_df, predicted_df, how='left', on='ncodpers')
high_frequency_products = "ind_cco_fin_ult1 ind_cno_fin_ult1 ind_ecue_fin_ult1 " \
                          "ind_ecue_fin_ult1 ind_nomina_ult1 ind_nom_pens_ult1 " \
                          "ind_recibo_ult1"
null_list = np.where(merged_df.added_products.isnull().values==1)[0]
merged_df.ix[null_list[0], 'added_products'] = high_frequency_products
file_name = '../output/sub_161217_ens1.csv'
merged_df.to_csv(file_name, index=False)
