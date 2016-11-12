# get the most frequent added products
import sqlite3 as sql
import pandas as pd
import numpy as np


connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)


train_total_count = pd.read_sql("select count(*) from santander_train", santanderCon)

product_used_df = pd.read_sql("select new_product.* from santander_train new_product where " +
                              "new_product.fecha_dato = '2015-04-28' and new_product.ncodpers " +
                              "in (select ncodpers from santander_train where fecha_dato = '2016-03-28') order by ncodpers", santanderCon)

product_used_old_df = pd.read_sql("select new_product.* from santander_train new_product where " +
                              "new_product.fecha_dato = '2015-03-28' and new_product.ncodpers " +
                              "in (select ncodpers from santander_train where fecha_dato = '2016-04-28') order by ncodpers", santanderCon)

diff_df = product_used_df.ix[:, 25:49] - product_used_old_df.ix[:, 25:49]
added_df = diff_df[diff_df == 1]
sum_df = added_df.sum(axis = 0)
print sum_df
