# Generate hyper-classes through ramdom trail.
# The total number of hyper-classes might vary, but at least 270 out of 286 combos are covered
import pandas as pd
import sqlite3 as sql
from tqdm import tqdm
import numpy as np
import random
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
del unique_df['index']
hclass_set = np.zeros([25, 16])
sixteen_one = np.ones([1, 16])

uncovered = unique_df.shape[0]+1
last_covered_count = 0
i = 0
end_loop = 0
while last_covered_count<270:
    # generate a random hyper class
    hclass_set[i][random.sample(range(16), 7)] = 1
    # test if the hyper class covers more than 20 unique combos
    uncovered_this = uncovered
    # compute how many new combos are covered
    covered_count = 0
    for j in range(0, unique_df.shape[0]):
        if max(np.dot(unique_df.ix[j].values, hclass_set.transpose())) >= sum(unique_df.ix[j]):
            covered_count += 1
    if covered_count - last_covered_count >= min(15 - i/2, 286-last_covered_count):
        i += 1
        last_covered_count = covered_count
        print hclass_set
        print covered_count
    else:
        hclass_set[i] = np.zeros(16)

# count number of effective rows
hclass_set = np.delete(hclass_set, range(i, 25), 0)
np.savetxt("../input/random_class_set1.csv", hclass_set, delimiter=",")
