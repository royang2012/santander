# generate prediction scores based on xgboost tree results. The score will be computed
# using the evaluation fucntion on Kaggle.com. Products added between 2016-05-28 and
# 2016-04-28 will be used as the true added products
import sqlite3 as sql
import pandas as pd
from tqdm import tqdm
import numpy as np

connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)
sql_select_truth = "select * from added1605"
truth_df = pd.read_sql(sql_select_truth, santanderCon)

predicted_df = pd.read_csv('../input/prediction.csv')

score = 0
if predicted_df.shape[0] == truth_df.shape[0]:
    for i in range(0, truth_df.shape[0]):
        real_array = truth_df.ix[i, 1:-2].values
        predicted_array = predicted_df.ix[i, 1:-2].astype('bool').values
        score += sum(real_array[predicted_array])/truth_df.ix[i, -2]
else:
    print "Results shape does not match!"

score /= predicted_df.shape[0]
print score
