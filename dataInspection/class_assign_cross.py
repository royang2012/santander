# read the randomly generated classes from HD and assign the original 24D output vector
# to the hyper-class.
# As there are usually more than one hyper-class that covers the output vector, a random
# choice is made between all hyper-classes that could cover the output to make the result
# more 'uniform'
import pandas as pd
import sqlite3 as sql
from tqdm import tqdm
import numpy as np
import random
connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)
select_train_out = "select * from santander_out_train"
train_out_df = pd.read_sql(select_train_out, santanderCon)
select_vali_out = "select * from santander_out_vali"
vali_out_df = pd.read_sql(select_vali_out, santanderCon)
del train_out_df['index']
del vali_out_df['index']
def deleteProducts(input_df):
    del input_df['ind_ahor_fin_ult1']
    del input_df['ind_aval_fin_ult1']
    del input_df['ind_cder_fin_ult1']
    del input_df['ind_deco_fin_ult1']
    del input_df['ind_deme_fin_ult1']
    del input_df['ind_hip_fin_ult1']
    del input_df['ind_pres_fin_ult1']
    del input_df['ind_viv_fin_ult1']
deleteProducts(train_out_df)
deleteProducts(vali_out_df)

class_set = np.loadtxt('../input/random_class_set1.csv', delimiter=',')

picked_class_train = np.zeros(train_out_df.shape[0])
for i in tqdm(range(0, train_out_df.shape[0])):
    class_score = np.dot(train_out_df.ix[i].values, class_set.transpose())
    max_positions = np.argwhere(class_score == np.amax(class_score))
    picked_class_train[i] = random.choice(max_positions)[0]

np.savetxt("../input/class_out.csv", picked_class_train, delimiter=",")
