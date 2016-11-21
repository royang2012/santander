import sqlite3 as sql

import pandas as pd
import random
import numpy as np
from tqdm import tqdm


def get_product_dict():
    connectionPath = "../santander_data.db"
    santanderCon = sql.connect(connectionPath)

    selectSql = " select * from santander_train limit 1"
    df = pd.read_sql(selectSql, santanderCon)
    productNames = df.ix[:, 25:49].columns
    i = 1
    product_dict = {}
    for product in productNames:
        product_dict[i] = product
        i += 1
    santanderCon.close()
    return product_dict


def get_products_for_users_in_test(date):
    connectionPath = "../santander_data.db"
    santanderCon = sql.connect(connectionPath)
    productValues = ",tr.".join(get_product_dict().values())  # construct comma separated string with all products
    selectUserProduct_sql = "select tr.ncodpers, tr." + productValues + " from santander_train tr inner join santander_test te " \
                                                                        "on tr.ncodpers=te.ncodpers " \
                                                                        "where tr.fecha_dato='" + date + "' limit 10"
    selectUserProduct = pd.read_sql(selectUserProduct_sql, santanderCon)
    santanderCon.close()
    return selectUserProduct


def get_used_product_freq():
    connectionPath = "../santander_data.db"
    santanderCon = sql.connect(connectionPath)

    # construct product count string
    productCounts_sql = ",".join(map(lambda p: "sum(" + p + ") as " + p, get_product_dict().values()))
    sqlstatement = "select " + productCounts_sql + " from santander_train"
    product_total_count = pd.read_sql(sqlstatement, santanderCon).transpose()
    santanderCon.close()
    product_total_count['count'] = product_total_count[0]
    del product_total_count[0]
    product_total_count.index.name = 'product'
    total_product = product_total_count['count'].sum()
    sorted_product_count = product_total_count.sort_values(by='count', axis=0, ascending=False)
    return sorted_product_count / total_product


def recommend_products_for_user(product_freq, user_current_products, maximum):
    added_products = []
    print "Generate recommendation for ncodpers {}".format(user_current_products['ncodpers'])
    for product in get_product_dict().values():
        if user_current_products[product] == 0:
            rn = random.random()
            if rn < product_freq[product]:
                added_products.append(product)
        if len(added_products) == maximum:
            break
    return added_products


# df['price'] = df.apply(lambda row: valuation_formula(row['x'], row['y']), axis=1)

def recommend_products(product_dict, product_freq, all_users_current_products, maximum):
    user_added = pd.DataFrame(columns=['ncodpers', 'added_products'])
    # for index, row in all_users_current_products.iterrows():
    # user_added.append(row.ncodpers, ','.join(recommend_products(product_freq, row, maximum)))

    for i in tqdm(range(0, all_users_current_products.shape[0])):
        # print all_users_current_products.ix[i]
        # print ','.join(recommend_products(product_freq, all_users_current_products.ix[i], maximum))
        added_products = []
        for product in product_dict.values():
            if np.isclose(all_users_current_products.ix[i].product, 0.0):
                rn = random.random()
                print "probability to be added: {}, actually {}".format(product_freq.ix[product], rn)
                if rn < product_freq.ix[product]*10:
                    added_products.append(product)
            if len(added_products) == maximum:
                break
        print added_products
        user_added.append([all_users_current_products.ix[i].ncodpers, ",".join(added_products)])
        # user_added.append(
        # {'ncodpers': all_users_current_products.ix[i].ncodpers, 'added_products': ",".join(added_products)})

    return user_added


product_dict = get_product_dict()
product_freq = get_used_product_freq()
user_products = get_products_for_users_in_test('2016-05-28')
maximus_added_products = 7

print recommend_products(product_dict, product_freq, user_products, maximus_added_products)
