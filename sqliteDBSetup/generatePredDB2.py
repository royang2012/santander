import sqlite3 as sql
import pandas as pd
from tqdm import tqdm

connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)

# table_name = 'santander_train'
# drop_statement = "drop table if exists " + table_name
# santanderCon.execute(drop_statement)

# select_statement = "select * from santander_train order by ncodpers, fecha_dato DESC"
# select_statement = "select * from santander_train where fecha_dato in " \
#                    "('2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28') " \
# #                    "order by ncodpers, fecha_dato DESC"
# select_statement = "select st.*, st2.total from santander_train st, " \
#                    "(select ncodpers, count(*) as total from santander_train where fecha_dato in " \
#                    "('2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28') " \
#                    "group by ncodpers having count(*) < 6) as st2 " \
#                    "where st.ncodpers=st2.ncodpers order by st.ncodpers, st.fecha_dato DESC"
# train_df = pd.read_sql(select_statement, santanderCon)

# select_statement6 = "select st.*, st2.total from santander_train st, " \
#                     "(select ncodpers, count(*) as total from santander_train where fecha_dato in " \
#                     "('2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28') " \
#                     "group by ncodpers ) as st2 " \
#                     "where st.ncodpers=st2.ncodpers " \
#                     "and st.fecha_dato in " \
#                     "('2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28') " \
#                     "order by st.ncodpers, st.fecha_dato DESC"
# train_df = pd.read_sql(select_statement6, santanderCon)
train_df = pd.read_csv('../input/train_batch_incom.csv')
train_df2 = train_df.sort(['ncodpers', 'fecha_dato'], ascending=[True, False])

# train_df.drop(train_df.index[8], inplace=1)
# train_df.drop(train_df.index[18], inplace=1)
# train_df.set_value(8, 'fecha_dato', '123')
# train_df.set_value(18, 'fecha_dato', '123')


# train_list = []
date_list = ['2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28', '2015-11-28',
             '2015-10-28', '2015-09-28', '2015-08-28', '2015-07-28', '2015-06-28', '2015-05-28', '2015-04-28',
             '2015-03-28', '2015-02-28', '2015-01-28']

# last_idx = 15888
# month_count = 0
# date_count = 1  # current index in data_list
# over7 = 0  # tag if there are over 7 months continuous data
# appended = 0
patch_list = []

# while loop, manipulate the group with size=count; add rows to patch_list; then row+=count
last_idx = 15889
month_count = 0
date_count = 1  # current index in data_list
over6 = 0   # tag if there are over 6 months continuous data
appended = 0
singular = 0
current_customer = train_df.ix[0:5]
for i in tqdm(range(6, train_df.shape[0])):
    if train_df.ix[i].ncodpers != last_idx:
        # current_customer.to_sql('santander_train7', santanderCon, if_exists='append')
        patch_list.append(current_customer)
        month_count = 0
        singular = 0
        current_month = date_list[month_count]
        if train_df.ix[i].fecha_dato == current_month:
            current_customer.ix[month_count] = train_df.ix[i]
        else:
            singular = 1
    else:
        if singular == 0:
            current_month = date_list[month_count]
            if train_df.ix[i].fecha_dato == current_month:
                current_customer.ix[month_count] = train_df.ix[i]
            else:
                for j in range(month_count, 6):
                    current_customer.ix[j] = train_df.ix[i]
                    current_customer.ix[j, 'fecha_dato'] = date_list[j]
                singular = 1
patch_list.append(current_customer)

# patch_list_df = pd.concat(patch_list)
# print patch_list_df[0]
# pred_df = pd.concat([patch_list_df, train_df])
# pred_df.sort(['ncodpers', 'fecha_dato'], ascending=[True, False])

# del pred_df['total']
# del pred_df['index']

# print patch_list[9].ix['fecha_dato']
