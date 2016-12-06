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

select_statement6 = "select st.*, st2.total as total from santander_train st, " \
                    "(select ncodpers, count(*) as total from santander_train where fecha_dato in " \
                    "('2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28') " \
                    "group by ncodpers ) as st2, " \
                    "(select ncodpers from santander_train where fecha_dato = '2016-05-28') as st3 " \
                    "where st.ncodpers=st2.ncodpers and st.ncodpers = st3.ncodpers " \
                    "and st.fecha_dato in " \
                    "('2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28') " \
                    "order by st.ncodpers, st.fecha_dato DESC"
train_df = pd.read_sql(select_statement6, santanderCon)
# train_df = pd.read_csv('../input/train_batch_incom.csv')
# train_df.drop(train_df.index[8], inplace=1)
# train_df.drop(train_df.index[18], inplace=1)
# train_df.set_value(8, 'fecha_dato', '123')
# train_df.set_value(18, 'fecha_dato', '123')


# train_list = []
# date_list = ['2016-05-28', '2016-04-28', '2016-03-28', '2016-02-28', '2016-01-28', '2015-12-28', '2015-11-28',
#              '2015-10-28', '2015-09-28', '2015-08-28', '2015-07-28', '2015-06-28', '2015-05-28', '2015-04-28',
#              '2015-03-28', '2015-02-28', '2015-01-28']
date_list = [train_df.ix[0, 'fecha_dato'], train_df.ix[1, 'fecha_dato'], train_df.ix[2, 'fecha_dato'],
             train_df.ix[3, 'fecha_dato'], train_df.ix[4, 'fecha_dato'], train_df.ix[5, 'fecha_dato']]
# last_idx = 15888
# month_count = 0
# date_count = 1  # current index in data_list
# over7 = 0  # tag if there are over 7 months continuous data
# appended = 0
patch_list = []

# while loop, manipulate the group with size=count; add rows to patch_list; then row+=count
row = 0
setSize = 6
pad_df = train_df.ix[0:2].copy(deep=True)
while row < train_df.shape[0]:
    userSetSize = train_df.ix[row].total
    if userSetSize == 6:
        row += userSetSize
    else:
        processedCount = 0
        for i in range(0, setSize):
            # print "User Set Size : {}".format(userSetSize)
            # print "Current date: {}".format(date_list[i])
            # print "processed {}".format(processedCount)
            # print "Current row: {}".format(row)
            if processedCount == userSetSize:
                # print "Add ending patch: processed {}".format(processedCount)
                patch_df = train_df.ix[row - 1].copy(deep=True)
                # patch_df['fecha_dato'] = date_list[i]
                patch_list.append(patch_df)
            elif train_df.ix[row].fecha_dato != date_list[i]:
                # print train_df.ix[row].fecha_dato
                patch_df = train_df.ix[row - 1].copy(deep=True)
                # patch_df['fecha_dato'] = date_list[i]
                patch_list.append(patch_df)
            elif train_df.ix[row].fecha_dato == date_list[i]:
                # print train_df.ix[row].fecha_dato
                row += 1
                processedCount += 1
    if row%10000 == 0:
        print row

patch_list_df = pd.concat(patch_list,axis = 1).transpose()
# print patch_list_df[0]
pred_df = pd.concat([patch_list_df, train_df])
pred_df_s = pred_df.sort(['ncodpers', 'fecha_dato'], ascending=[True, False])
#
del pred_df_s['total']
del pred_df_s['index']
for i in range(pred_df_s.shape[1]):
    pred_df_s[pred_df_s.columns[i]] = pred_df_s[pred_df_s.columns[i]].astype(train_df.dtypes[i])
#
# print patch_list[9].ix['fecha_dato']
# pred_df_s.to_sql('santander_pred6', santanderCon, if_exists='append')
pred_df_s.to_csv('../input/pred6.csv', encoding='utf-8')

