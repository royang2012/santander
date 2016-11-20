import sqlite3 as sql
import pandas as pd
from tqdm import tqdm
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)

table_name = 'santander_train'
# drop_statement = "drop table if exists " + table_name
# santanderCon.execute(drop_statement)

select_statement = "select * from santander_train strain order by ncodpers, fecha_dato DESC"
train_df = pd.read_sql(select_statement, santanderCon)

# train_df.drop(train_df.index[8], inplace=1)
# train_df.drop(train_df.index[18], inplace=1)
# train_df.set_value(8, 'fecha_dato', '123')
# train_df.set_value(18, 'fecha_dato', '123')


# train_list = []
date_list = ['2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28', '2015-11-28',
             '2015-10-28', '2015-09-28', '2015-08-28', '2015-07-28', '2015-06-28', '2015-05-28', '2015-04-28',
             '2015-03-28', '2015-02-28', '2015-01-28']
last_idx = 15889
month_count = 0
date_count = 1  # current index in data_list
over7 = 0   # tag if there are over 7 months continuous data
appended = 0
for i in tqdm(range(1, train_df.shape[0])):
    if train_df.ix[i].ncodpers == last_idx:
        current_date = date_list[date_count]
        if train_df.ix[i].fecha_dato == current_date:
            month_count += 1
            if month_count == 6:
                over7 = 1
            if train_df.ix[i].fecha_dato == '2015-01-28':
                current_customer = train_df.ix[i-month_count:i, 1:]
                current_customer.to_sql('santander_train7', santanderCon, if_exists='append')
                appended = 1
        else:
            if over7 == 1:
                if appended == 0:
                    current_customer = train_df.ix[i-month_count-1:i-1, 1:]
                    current_customer.to_sql('santander_train7', santanderCon, if_exists='append')
                    appended = 1
    else:
        if over7 == 1 and appended == 0:
            current_customer = train_df.ix[i-month_count-1:i-1, 1:]
            current_customer.to_sql('santander_train7', santanderCon, if_exists='append')
        date_count = 0
        month_count = 0
        over7 = 0
        appended = 0
    last_idx = train_df.ix[i].ncodpers
    date_count += 1
# test_6_df = pd.concat(train_list)
#
# test_6_df.to_sql('santander_train7', santanderCon, if_exists='append')
