import sqlite3 as sql
import pandas as pd
from tqdm import tqdm
import numpy as np
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)
sqlStatement = "drop table if exists added1605"
santanderCon.execute(sqlStatement)
# read customer data from the last two months from the training set
select_statement = "select * from santander_train7 strain where fecha_dato = '2016-04-28' order by ncodpers"
train_df = pd.read_sql(select_statement, santanderCon)
select_statement_2 = "select * from santander_train7 strain where fecha_dato = '2016-03-28' order by ncodpers"
train_df2 = pd.read_sql(select_statement_2, santanderCon)

# compute the difference
diff_df = train_df.ix[:, 25:49] - train_df2.ix[:, 25:49]

# clean out redundant info
# del train_df['index']
# del train_df['fecha_dato']
# redundant_column_list = np.arange(1, 23)
# train_df.drop(train_df.columns[redundant_column_list], axis=1, inplace=True)

# find the added products and get rid of NANs
added_df = diff_df[diff_df == 1]
added_df = added_df.fillna(0)

for i in range(0, train_df.shape[0]):
    added_df.loc[i, 'sum'] = np.sum(added_df.ix[i, :])

added_df.loc[:,'ncodpers'] = train_df.ncodpers

added_df.to_sql('added1605', santanderCon, if_exists='append')
