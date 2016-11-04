import sqlite3 as sql
import pandas as pd
import pandas.io.sql as pd_sql

import datetime as dt

import matplotlib.pyplot as plt

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

count_by_city = pd.read_sql("select nomprov, count(*) as count "
                            + "from (select distinct ncodpers, nomprov from santander_train)"
                            + " group by nomprov limit 10", santanderCon)
count_by_city.loc[count_by_city.nomprov == "CORU\xc3\x91A, A", "nomprov"] = "CORUNA, A"
print count_by_city
plt.bar(count_by_city["nomprov"], count_by_city["count"], 1, color="tomato")
plt.xlabel("nomprov")
plt.ylabel("Count")
plt.title("Location Distribution")
plt.show()
