import sqlite3 as sql
import pandas as pd
import pandas.io.sql as pd_sql

import datetime as dt

# import matplotlib.pyplot as plt

connectionPath = "./santander_data.db"
santanderCon = sql.connect(connectionPath)
#
# count_by_age = pd.read_sql("select age, count(*) as count from (select distinct ncodpers, age from santander_train)" +
#                            " group by age", santanderCon)
# print count_by_age
# count_by_age["age"] = pd.to_numeric(count_by_age["age"], errors="coerce")
#
# width = 1
# plt.bar(count_by_age["age"], count_by_age["count"], width, color="tomato")
# plt.xlabel("Age")
# plt.ylabel("Count")
# plt.title("Age Distribution")
# plt.show()

# count_by_city = pd.read_sql("select nomprov, count(*) as count "
#                             + "from (select distinct ncodpers, nomprov from santander_train)"
#                             + " group by nomprov limit 10", santanderCon)
# count_by_city.loc[count_by_city.nomprov == "CORU\xc3\x91A, A", "nomprov"] = "CORUNA, A"
# print count_by_city
# plt.bar(count_by_city["nomprov"], count_by_city["count"], 1, color="tomato")
# plt.xlabel("nomprov")
# plt.ylabel("Count")
# plt.title("Location Distribution")
# plt.show()
#


# count_by_date_sql = " select fecha_dato, count(distinct ncodpers) as count from santander_train group by fecha_dato limit 100"
# count_by_date = pd.read_sql(count_by_date_sql, santanderCon)
# count_by_date["fecha_dato"] = pd.to_datetime(count_by_date["fecha_dato"], errors="coerce")
# print count_by_date
#
# width = 1
# # plt.bar(count_by_date["fecha_dato"], count_by_date["count"], width, color="tomato")
# plt.plot(count_by_date["fecha_dato"], count_by_date["count"])
#
# plt.xlabel("fecha_dato")
# plt.ylabel("Count")
# plt.title("User Count Distribution")
# plt.show()

train_9month_sql = " select ncodpers from santander_train where fecha_dato>='2015-09-28' and fecha_dato<='2016-05-28' " \
                   " group by ncodpers having count(*)==9"
# train_9month = pd.read_sql(train_9month_sql, santanderCon)
# print train_9month
#
# validate_sql = "select ncodpers, fecha_dato from santander_train where ncodpers='15890'"
# validate_df = pd.read_sql(validate_sql, santanderCon)
# print validate_df

validated_train_9months_sql = "select strain.* from santander_train strain inner join " \
                              " (" + train_9month_sql + ") as strainNine on strain.ncodpers = strainNine.ncodpers" \
                              " inner join (select distinct ncodpers from santander_test) as stest " \
                              " on strain.ncodpers = stest.ncodpers " \
                              " where strain.fecha_dato>='2015-09-28' and strain.fecha_dato<='2016-05-28'"
validated_train_9months = pd.read_sql(validated_train_9months_sql, santanderCon)
print validated_train_9months
validated_train_9months.to_csv("../output/train_9months.csv")
