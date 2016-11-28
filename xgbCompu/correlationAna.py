import sqlite3 as sql
import pandas as pd
import numpy as np
import dataPrepa as dP
import matplotlib.pyplot as plt
# read and prepare data
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)
df1 = pd.read_sql("select * from santander_train7 where fecha_dato = '2016-05-28' order by ncodpers", santanderCon)
df2 = pd.read_sql("select * from santander_train7 where fecha_dato = '2016-04-28' order by ncodpers", santanderCon)
df3 = pd.read_sql("select * from santander_train7 where fecha_dato = '2016-03-28' order by ncodpers", santanderCon)
df4 = pd.read_sql("select * from santander_train7 where fecha_dato = '2016-02-28' order by ncodpers", santanderCon)
diff_df1 = df1.ix[:, 25:49] - df2.ix[:, 25:49]
diff_df2 = df2.ix[:, 25:49] - df3.ix[:, 25:49]
diff_df3 = df3.ix[:, 25:49] - df4.ix[:, 25:49]
added_df1 = diff_df1[diff_df1 == 1]
added_df1 = added_df1.fillna(0)
added_df2 = diff_df2[diff_df2 == 1]
added_df2 = added_df2.fillna(0)
added_df3 = diff_df3[diff_df3 == 1]
added_df3 = added_df3.fillna(0)

# # # note! # # #
# lots of feature and 6 products are deleted!
dP.deleteProducts(added_df1)
dP.deleteProducts(added_df2)
dP.deleteProducts(added_df3)
# deleteProducts(added_df1)
# deleteProducts(added_df2)
# deleteProducts(added_df3)

# compute the correlation between added products
cor1 = np.dot(added_df1.transpose().values, added_df1.values)
cor2 = np.dot(added_df2.transpose().values, added_df2.values)
cor3 = np.dot(added_df3.transpose().values, added_df3.values)
cor_matrix = cor1 + cor2 + cor3
for i in range(0, cor_matrix.shape[0]):
    cor_matrix[i, i] = 0

print added_df1.columns
plt.matshow(cor_matrix, cmap=plt.cm.gray)
plt.show()
