# get the most frequent added products
import sqlite3 as sql
import pandas as pd
import numpy as np


connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)


train_total_count = pd.read_sql("select count(*) from santander_train", santanderCon)

df1 = pd.read_sql("select * from santander_train7 where fecha_dato = '2016-05-28' order by ncodpers", santanderCon)
df2 = pd.read_sql("select * from santander_train7 where fecha_dato = '2016-04-28' order by ncodpers", santanderCon)
df3 = pd.read_sql("select * from santander_train7 where fecha_dato = '2016-03-28' order by ncodpers", santanderCon)
df4 = pd.read_sql("select * from santander_train7 where fecha_dato = '2016-02-28' order by ncodpers", santanderCon)

diff_df1 = df1.ix[:, 25:49] - df2.ix[:, 25:49]
diff_df2 = df2.ix[:, 25:49] - df3.ix[:, 25:49]
diff_df3 = df3.ix[:, 25:49] - df4.ix[:, 25:49]

added_df1 = diff_df1[diff_df1 == 1]
added_df1 = added_df1.fillna(0)
added_df2 = diff_df2[diff_df2 == 1]
added_df2 = added_df2.fillna(0)
added_df3 = diff_df3[diff_df3 == 1]
added_df3 = added_df3.fillna(0)

sum_df1 = added_df1.sum(axis = 0)
sum_df2 = added_df2.sum(axis = 0)
sum_df3 = added_df3.sum(axis = 0)
con_df = pd.concat([sum_df1, sum_df2, sum_df3], axis=0)
sum_df = con_df.sum(axis=0)
print sum_df1
print sum_df
