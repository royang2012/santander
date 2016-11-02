import sqlite3 as sql

import pandas as pd
import pandas.io.sql as pd_sql
from sqlalchemy import create_engine  # database connection
import datetime as dt
from IPython.display import display

import plotly.plotly as py  # interactive graphing
from plotly.graph_objs import Bar, Scatter, Marker, Layout



def createTableWithCsvFile(csvFile, tableName, connectionPath, chunksize):
    j = 0
    index_start = 1
    con = sql.connect(connectionPath)
    con.text_factory = str
    sqlStatement = "drop table if exists " + tableName
    con.execute(sqlStatement)
    start = dt.datetime.now()
    for df in pd.read_csv(csvFile, chunksize=chunksize, iterator=True, encoding='utf-8'):
        df.index += index_start
        j += 1
        print '{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j * chunksize)
        df.to_sql(tableName, con, if_exists='append')
        index_start = df.index[-1] + 1
    con.commit()
    con.close()
    print 'Finished loading data from {} to table {}'.format(csvFile, tableName)


def createTableIndex(createIndexSql, connectionPath):
    con = sql.connect(connectionPath)
    con.execute(createIndexSql)
    con.close()
    print "created index: {}".format(createIndexSql)




# print out the tables defined in santander_data.db
# santanderCursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print(santanderCursor.fetchall())
#
# # print out table schema: column names of table and datatype
# meta = santanderCursor.execute("PRAGMA table_info('santander_train')")
# for r in meta:
#     print r
