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

class_set = np.loadtxt('../input/random_class_set1.csv', delimiter=',')

picked_class = np.zeros(output_df.shape[0])
for i in tqdm(range(0, output_df.shape[0])):
    class_score = np.dot(output_df.ix[i].values, class_set.transpose())
    max_positions = np.argwhere(class_score == np.amax(class_score))
    picked_class[i] = random.choice(max_positions)[0]

np.savetxt("../input/class_out.csv", picked_class, delimiter=",")
