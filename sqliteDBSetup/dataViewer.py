# View data stored in Database "./santander_data.db"
# tables in the database: santander_train.

import pandas as pd
import sqlite3 as sql

connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)

# train_data = pd.read_sql("select * from santander_train limit 5", santanderCon)
# print train_data
#
# train_data_count_by_userId = pd.read_sql("select count(distinct ncodpers ) from santander_train limit 5", santanderCon)
# print train_data_count_by_userId
#
# train_data_count_by_fetch_date = pd.read_sql("select count(distinct fecha_dato ) from santander_train limit 5", santanderCon)
# print train_data_count_by_fetch_date

# train_data_match_by_fetch_date = pd.read_sql("select count(*) from santander_train where ncodpers like '%234%' and fecha_dato='2015-01-28' ", santanderCon)
# print train_data_match_by_fetch_date
# now we can run more complicated queries using sql, or even create more useful indices for our use cases.

# order_by_idx_then_date = "select * from santander_train order by ncodpers, fecha_dato limit 50"
# print pd.read_sql(order_by_idx_then_date, santanderCon)

aggregate_by_idx = "select ncodpers, count(*) from santander_train group by ncodpers"
print pd.read_sql(aggregate_by_idx, santanderCon)
