# find and save those entries with 6 months data and added products at the 7's month.
# Features are extracted and stored in table 'santander_feature6' while the output
# is store in table 'santander_out' there is no index on the output table, just ordered
# by ncodpers and fecha_dato descendingly
import sqlite3 as sql
import pandas as pd
from tqdm import tqdm
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)

# table_name = 'santander_train'
# drop_statement = "drop table if exists " + table_name
# santanderCon.execute(drop_statement)

select_statement = "select * from santander_train7 order by ncodpers, fecha_dato DESC"
train_df = pd.read_sql(select_statement, santanderCon)

# train_df.drop(train_df.index[8], inplace=1)
# train_df.drop(train_df.index[18], inplace=1)
# train_df.set_value(8, 'fecha_dato', '123')
# train_df.set_value(18, 'fecha_dato', '123')


product_col_list = train_df.columns[25:49]
contain_one = lambda x: 1 if (1 in x[product_col_list].values) else 0
train_df['added'] = (train_df[product_col_list] - train_df[product_col_list].shift(-1)).apply(contain_one, axis=1)
# train_list = []
# date_list = ['2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28', '2015-11-28',
#              '2015-10-28', '2015-09-28', '2015-08-28', '2015-07-28', '2015-06-28', '2015-05-28', '2015-04-28',
#              '2015-03-28', '2015-02-28', '2015-01-28']
# last_idx = 15889
# month_count = 0
# date_count = 1  # current index in data_list
# over7 = 0   # tag if there are over 7 months continuous data
# appended = 0
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
