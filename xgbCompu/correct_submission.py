# try to use the correct submission generation function
import numpy as np

def deleteProducts(input_df):
    # del input_df['ind_ahor_fin_ult1']
    # del input_df['ind_aval_fin_ult1']
    del input_df['ind_cder_fin_ult1']
    # del input_df['ind_deco_fin_ult1']
    # del input_df['ind_deme_fin_ult1']
    del input_df['ind_hip_fin_ult1']
    del input_df['ind_pres_fin_ult1']
    del input_df['ind_viv_fin_ult1']

preds = np.loadtxt('../input/pred_1207.csv', delimiter=",")
