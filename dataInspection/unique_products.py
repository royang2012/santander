# find the unique combo of added products
# try to cover all cases with fewer 7 point hyper classes
import pandas as pd
import sqlite3 as sql
from tqdm import tqdm
import numpy as np

connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)
select_statement = "select * from santander_out"
output_df = pd.read_sql(select_statement, santanderCon)
del output_df['index']
def deleteProducts(input_df):
    del input_df['ind_ahor_fin_ult1']
    del input_df['ind_aval_fin_ult1']
    del input_df['ind_cder_fin_ult1']
    del input_df['ind_deco_fin_ult1']
    del input_df['ind_deme_fin_ult1']
    del input_df['ind_hip_fin_ult1']
    del input_df['ind_pres_fin_ult1']
    del input_df['ind_viv_fin_ult1']
deleteProducts(output_df)
# how many unique combos are there
unique_df = output_df.drop_duplicates()
print unique_df.shape
unique_df.reset_index(inplace=True)

# generate a set of hyper class. each hyper class should contain seven 1's
# from the analysis of covariance, we know that
# the 2nd, 14th and 15th products has some correlation between them, so we first
# set three ones there.
# hclass_set = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
#                       [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
#                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
#                       [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
#                       [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
#                       [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1]])
hclass_set = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                       [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0]])

singular = []
not_in = 0
for i in range(0, unique_df.shape[0]):
    if max(np.dot(unique_df.ix[i, 1:].values, hclass_set.transpose())) < sum(unique_df.ix[i, 1:]):
        not_in += 1
        singular.append(unique_df.ix[i])

print not_in
singular_df = pd.concat(singular, axis=1).transpose()
singular_df.reset_index(inplace=True)
