# find and save those entries with 6 months data and added products at the 7's month.
# Features are extracted and stored in table 'santander_feature6' while the output
# is store in table 'santander_out' there is no index on the output table, just ordered
# by ncodpers and fecha_dato descendingly
import sqlite3 as sql
import pandas as pd
from tqdm import tqdm
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)

select_statement = "select * from santander_train7 order by ncodpers, fecha_dato DESC"
train_df = pd.read_sql(select_statement, santanderCon)

product_col_list = train_df.columns[25:49]
contain_one = lambda x: 1 if (1 in x[product_col_list].values) else 0
# tell if the customer added any product in the month
train_df['added'] = (train_df[product_col_list] - train_df[product_col_list].shift(-1)).apply(contain_one, axis=1)



feature_list = []
output_list = []
i = 0
# for i in tqdm(range(1, train_df.shape[0]-7)):
while i <= train_df.shape[0]-7:
    if train_df.ix[i, 'ncodpers'] == train_df.ix[i+6, 'ncodpers']:
        if train_df.ix[i, 'added'] == 1:
            feature_list.append(train_df.ix[i+1:i+6, 0:49])
            output_list.append(train_df.ix[i, product_col_list] - train_df.ix[i+1, product_col_list])
        i += 1
    else:
        i += 6

feature_df = pd.concat(feature_list)
output_df = pd.concat(output_list, axis = 1).transpose()
#
# test_6_df.to_sql('santander_train7', santanderCon, if_exists='append')
feature_df.to_sql('santander_feature6', santanderCon, if_exists='append')
output_df2 = output_df[output_df == 1]
output_df2 = output_df2.fillna(0)
output_df3 = output_df2.astype(int)
output_df3.to_sql('santander_out', santanderCon, if_exists='append')
