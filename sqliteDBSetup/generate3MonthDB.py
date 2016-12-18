# find and save those entries with 6 months data and added products at the 7's month.
# Features are extracted and stored in table 'santander_feature6' while the output
# is store in table 'santander_out' there is no index on the output table, just ordered
# by ncodpers and fecha_dato descendingly
import sqlite3 as sql
import pandas as pd
from tqdm import tqdm
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)

select_statement = "select * from santander_train order by ncodpers, fecha_dato DESC"
train_df = pd.read_sql(select_statement, santanderCon)

product_col_list = train_df.columns[25:49]
contain_one = lambda x: 1 if (1 in x[product_col_list].values) else 0
# tell if the customer added any product in the month
train_df['added'] = (train_df[product_col_list] - train_df[product_col_list].shift(-1)).apply(contain_one, axis=1)

vali_feature_list = []
vali_output_list = []
train_feature_list = []
train_output_list = []
i = 0
# for i in tqdm(range(1, train_df.shape[0]-7)):
while i <= train_df.shape[0]-4:
    if train_df.ix[i, 'ncodpers'] == train_df.ix[i+2, 'ncodpers']:
        if train_df.ix[i, 'added'] == 1:
            # tell if the start date is 05-28, which is reserved for validation
            if train_df.ix[i, 'fecha_dato'] == '2016-05-28':
                # flag = 1
                # current_customer = train_df.ix[i+1:i+6, 0:49]
                # current_customer.to_sql('santander_feature6_vali', santanderCon, if_exists='append')
                # current_out = train_df.ix[i, product_col_list] - train_df.ix[i+1, product_col_list]
                # current_out.to_sql('santander_out_vali', santanderCon, if_exists='append')
                vali_feature_list.append(train_df.ix[i+1:i+2, 0:49])
                vali_output_list.append(train_df.ix[i, product_col_list] - train_df.ix[i+1, product_col_list])
            else:
                # current_customer = train_df.ix[i+1:i+6, 0:49]
                # current_customer.to_sql('santander_feature6_train', santanderCon, if_exists='append')
                # current_out = train_df.ix[i, product_col_list] - train_df.ix[i+1, product_col_list]
                # current_out.to_sql('santander_out_train', santanderCon, if_exists='append')
                #
                train_feature_list.append(train_df.ix[i+1:i+2, 0:49])
                train_output_list.append(train_df.ix[i, product_col_list] - train_df.ix[i+1, product_col_list])
    i += 1

vali_feature_df = pd.concat(vali_feature_list)
vali_output_df = pd.concat(vali_output_list, axis = 1).transpose()
train_feature_df = pd.concat(train_feature_list)
train_output_df = pd.concat(train_output_list, axis = 1).transpose()

#
# test_6_df.to_sql('santander_train7', santanderCon, if_exists='append')
vali_feature_df.to_sql('santander_feature2_vali', santanderCon, if_exists='append')
output_df2 = vali_output_df[vali_output_df == 1]
output_df2 = output_df2.fillna(0)
output_df3 = output_df2.astype(int)
output_df3.to_sql('santander_out2_vali', santanderCon, if_exists='append')

train_feature_df.to_sql('santander_feature2_train', santanderCon, if_exists='append')
output_df4 = train_output_df[train_output_df == 1]
output_df4 = output_df4.fillna(0)
output_df5 = output_df4.astype(int)
output_df5.to_sql('santander_out2_train', santanderCon, if_exists='append')
