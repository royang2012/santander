# generate a set of test feature bases where for each customer, there will be at least 12 months history
import sqlite3 as sql
import pandas as pd
from tqdm import tqdm

connectionPath = "./santander_data.db"
santanderCon = sql.connect(connectionPath)

sql_command = "select strain.* from santander_train strain where strain.ncodpers in (select distinct ncodpers from santander_test) " \
              " order by strain.ncodpers, strain.fecha_dato DESC"
train_df = pd.read_sql(sql_command, santanderCon)

train_list = []
date_list = ['2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28', '2015-11-28',
             '2015-10-28', '2015-09-28', '2015-08-28', '2015-07-28', '2015-06-28', '2015-05-28', '2015-04-28',
             '2015-03-28', '2015-02-28', '2015-01-28']
last_idx = 15889
month_count = 0
date_count = 1
for i in tqdm(range(1, train_df.shape[0])):
    if train_df.ix[i].ncodpers == last_idx:
        current_date = date_list[date_count]
        if train_df.ix[i].fecha_dato == current_date:
            month_count += 1
            if month_count == 8:
                # month_count = 0
                if train_df.ix[i-8].fecha_dato == date_list[0]:
                    train_list.append(train_df.ix[i-8:i])
    else:
        date_count = 0
        month_count = 0
    last_idx = train_df.ix[i].ncodpers
    date_count += 1
test_9_df = pd.concat(train_list)
# print test_9_df
# train_17_df.to_csv('./input/sorted_17_train.csv', index=False)
