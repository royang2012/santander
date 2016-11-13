import pandas as pd
import sqlite3 as sql
from sklearn.datasets import dump_svmlight_file

connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)

pre_month_df = pd.read_sql("select new_product.* from santander_train new_product where " +
                            "new_product.fecha_dato = '2015-03-28' and new_product.ncodpers " +
                            "in (select ncodpers from santander_train where fecha_dato = '2016-04-28') " +
                            "order by ncodpers", santanderCon)

cur_month_df = pd.read_sql("select new_product.* from santander_train new_product where " +
                            "new_product.fecha_dato = '2015-04-28' and new_product.ncodpers " +
                            "in (select ncodpers from santander_train where fecha_dato = '2016-03-28') " +
                            "order by ncodpers", santanderCon)

next_month_df = pd.read_sql()
