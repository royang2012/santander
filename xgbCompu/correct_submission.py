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

# kaggle submission generation
target_cols = np.array(train_fea_df.columns[16:])
preds_s = np.argsort(preds, axis=1)
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
file_name = '../output/sub_161211_their_sub_binary_xgb.csv'
merged_df.to_csv(file_name, index=False)
