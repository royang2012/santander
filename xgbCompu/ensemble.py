import pandas as pd
from tqdm import tqdm
readin1 = pd.read_csv('../output/sub_161207.csv')
readin2 = pd.read_csv('C:\Users\Royan\Downloads\sample_submission.csv')

split_products = lambda x: str.split(x)
pred_products = readin1.added_products.apply(split_products)
pop_products = readin2.added_products.apply(split_products)

in_pop = set(pop_products[0])
final_prediction = []
for i in tqdm(range(0, readin1.shape[0])):
    in_pred = set(pred_products[i])
    only_in_pred = in_pred - in_pop
    if len(only_in_pred) == 1:
        pop_products[i].append(list(only_in_pred)[0])
    else:
        del pop_products[i][-1]
        pop_products[i].append(list(only_in_pred)[0])
        pop_products[i].append(list(only_in_pred)[1])
    final_prediction.append(" ".join(pop_products[i]))

df_p = pd.DataFrame(final_prediction, columns=['added_products'])
df_i = pd.DataFrame(readin1.ncodpers.values, columns=['ncodpers'])
df_pred = pd.concat([df_i, df_p], axis=1)
df_pred.to_csv('../output/sub_161207_ensemble.csv', index=False)
