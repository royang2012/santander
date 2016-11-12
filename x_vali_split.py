# generate the validation set.
import sqlite3 as sql
import pandas as pd

connectionPath = "./santander_data.db"
santanderCon = sql.connect(connectionPath)

sql_command = "select strain.* from santander_train strain where strain.ncodpers in (select distinct ncodpers from santander_test) " \
              " and strain.fecha_dato = '2015-05-28' order by strain.ncodpers"
train_df = pd.read_sql(sql_command, santanderCon)

cols = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

train_df.drop(train_df.columns[cols], axis=1)
train_df.to_csv('./input/0528product.csv', encoding='utf-8')
