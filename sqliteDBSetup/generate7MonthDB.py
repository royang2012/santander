import sqlite3 as sql
import pandas as pd
from tqdm import tqdm
connectionPath = "./santander_data.db"
santanderCon = sql.connect(connectionPath)

table_name = 'santander_train'
# drop_statement = "drop table if exists " + table_name
# santanderCon.execute(drop_statement)

select_statement = "select * from santander_train strain order by ncodpers, fecha_dato DESC"
train_df = pd.read_sql(select_statement, santanderCon)

train_list = []
date_list = ['2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28', '2015-11-28']

last_idx = 15889
month_count = 0
date_count = 1
for i in tqdm(range(1, train_df.shape[0])):
    if train_df.ix[i].ncodpers == last_idx:
        current_date = date_list[date_count]
        if train_df.ix[i].fecha_dato == current_date:
            month_count += 1
            if month_count == 6:
                # month_count = 0
                if train_df.ix[i-6].fecha_dato == date_list[0]:
                    train_list.append(train_df.ix[i-6:i])
    else:
        date_count = 0
        month_count = 0
    last_idx = train_df.ix[i].ncodpers
    date_count += 1
test_6_df = pd.concat(train_list)

test_6_df.to_sql('santander_train7', santanderCon, if_exists='append')
