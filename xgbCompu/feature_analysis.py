import pandas as pd
import xgboost as xgb
import operator
from matplotlib import pylab as plt
import sqlite3 as sql
import numpy as np


def preprocess_data():
    connectionPath = "../santander_data.db"
    santanderCon = sql.connect(connectionPath)

    pre_month_df = pd.read_sql("select new_product.* from santander_train new_product where " +
                               "new_product.fecha_dato = '2016-03-28' and new_product.ncodpers " +
                               "in (select ncodpers from santander_train where fecha_dato = '2016-04-28') " +
                               "order by ncodpers", santanderCon)

    cur_month_df = pd.read_sql("select new_product.* from santander_train new_product where " +
                               "new_product.fecha_dato = '2016-04-28' and new_product.ncodpers " +
                               "in (select ncodpers from santander_train where fecha_dato = '2016-03-28') " +
                               "order by ncodpers", santanderCon)

    length = pre_month_df.shape[0]

    diff_df = pre_month_df.ix[:, 25:49] - cur_month_df.ix[:, 25:49]
    added_df = diff_df[diff_df == 1]

    # delete unwanted columns
    del pre_month_df['fecha_alta']
    del pre_month_df['ult_fec_cli_1t']
    del pre_month_df['canal_entrada']
    del pre_month_df['nomprov']
    # force two columns to be numerical
    pre_month_df['age'] = pd.to_numeric(pre_month_df['age'], errors='coerce')
    pre_month_df['antiguedad'] = pd.to_numeric(pre_month_df['antiguedad'], errors='coerce')

    # find the data type for each column
    dtype_list = pre_month_df.dtypes
    numerical_list = []
    dummy_list = []
    num_df = []
    # separate numerical and non-numerical columns
    for i in range(3, pre_month_df.shape[1]):
        if (dtype_list[i] == 'object'):
            dummy_list.append(pre_month_df.columns[i])
        else:
            numerical_list.append(pre_month_df.columns[i])

    # convert obejcts to One-hot-vector and combine the features
    num_df = pre_month_df[numerical_list]
    num_df = pd.concat([num_df, pd.get_dummies(pre_month_df[dummy_list])], axis=1)

    # use the first 80% of customers for training and the last 20% for validation
    train_data = num_df.ix[0:length * 0.8, 0:].values
    test_data = num_df.ix[length * 0.8:, 0:].values

    train_X = train_data
    train_Y = np.nan_to_num(added_df.ix[0:length * 0.8, 6].values)

    test_X = test_data
    test_Y = np.nan_to_num(added_df.ix[length * 0.8:, 6].values)

    return num_df.columns, train_X, train_Y, test_X, test_Y


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


def get_data():
    train = pd.read_csv("../input/train.csv")

    features = list(train.columns[2:])

    y_train = train.Hazard

    for feat in train.select_dtypes(include=['object']).columns:
        m = train.groupby([feat])['Hazard'].mean()
        train[feat].replace(m, inplace=True)

    x_train = train[features]

    return features, x_train, y_train


features, train_X, train_Y, test_X, test_Y = preprocess_data()
ceate_feature_map(features)

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
xgb_params = {"objective": "reg:linear", "eta": 0.01, "max_depth": 8, "seed": 42, "silent": 1}
num_rounds = 10

gbdt = xgb.train(xgb_params, xg_train, num_rounds, watchlist)
# get prediction
pred = gbdt.predict(xg_test);

print ('predicting, classification error=%f' % (
    sum(int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))))
print ('all zero guess error=%f' % (np.sum(test_Y) / test_Y.shape[0]))

importance = gbdt.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')
