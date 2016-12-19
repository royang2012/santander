import pandas as pd
import numpy as np
from tqdm import tqdm
test_df = pd.read_csv('./input/test_ver2.csv')
for i in range(0, 22):
    del test_df[test_df.columns[2]]
del test_df['fecha_dato']
train_df = pd.read_csv('./input/train_ver2.csv')
product_col_list = train_df.columns[25:49]
contain_one = lambda x: 1 if (1 in x[product_col_list].values) else 0
# tell if the customer added any product in the month
train_df['added'] = (train_df[product_col_list] - train_df[product_col_list].shift(-1)).apply(contain_one, axis=1)
train_df.sort_values(['ncodpers', 'fecha_dato'], ascending=[True, False], inplace =True)
train_df.reset_index(inplace=True)
vali_feature_list = []
vali_output_list = []
train_feature_list = []
train_output_list = []
for i in range(0, train_df.shape[0]):
    if train_df.ix[i, 'added'] == 1:
        if train_df.ix[i+2, 'ncodpers'] == train_df.ix[i, 'ncodpers']:
            if train_df.ix[i, 'fecha_dato'] == '2016-05-28':
                # flag = 1
                # current_customer = train_df.ix[i+1:i+6, 0:49]
                # current_customer.to_sql('santander_feature6_vali', santanderCon, if_exists='append')
                # current_out = train_df.ix[i, product_col_list] - train_df.ix[i+1, product_col_list]
                # current_out.to_sql('santander_out_vali', santanderCon, if_exists='append')
                vali_feature_list.append(train_df.ix[i + 1:i + 2, 0:49])
                vali_output_list.append(train_df.ix[i, product_col_list] - train_df.ix[i + 1, product_col_list])
            else:
                # current_customer = train_df.ix[i+1:i+6, 0:49]
                # current_customer.to_sql('santander_feature6_train', santanderCon, if_exists='append')
                # current_out = train_df.ix[i, product_col_list] - train_df.ix[i+1, product_col_list]
                # current_out.to_sql('santander_out_train', santanderCon, if_exists='append')
                #
                train_feature_list.append(train_df.ix[i + 1:i + 2, 0:49])
                train_output_list.append(train_df.ix[i, product_col_list] - train_df.ix[i + 1, product_col_list])


vali_feature_df = pd.concat(vali_feature_list)
vali_output_df = pd.concat(vali_output_list, axis=1).transpose()
train_feature_df = pd.concat(train_feature_list)
train_output_df = pd.concat(train_output_list, axis=1).transpose()

        #
# test_6_df.to_sql('santander_train7', santanderCon, if_exists='append')
vali_feature_df.to_csv('./input/santander_feature2_vali.csv')
output_df2 = vali_output_df[vali_output_df == 1]
output_df2 = output_df2.fillna(0)
output_df3 = output_df2.astype(int)
output_df3.to_sql('./input/santander_out2_vali.csv')

train_feature_df.to_sql('./input/santander_feature2_train.csv')
output_df4 = train_output_df[train_output_df == 1]
output_df4 = output_df4.fillna(0)
output_df5 = output_df4.astype(int)
output_df5.to_sql('./input/santander_out2_train.csv')

