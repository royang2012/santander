import pandas as pd

training_df = pd.read_csv('./input/train_ver2.csv')

debug_batch_df = training_df[(training_df.ncodpers==15889)]
debug_batch_df1 = training_df[(training_df.ncodpers==15890)]
debug_batch_df2 = training_df[(training_df.ncodpers==15892)]
debug_batch_df3 = training_df[(training_df.ncodpers==15893)]
debug_batch_df4 = training_df[(training_df.ncodpers==15894)]
debug_batch_df5 = training_df[(training_df.ncodpers==15895)]
debug_batch_df6 = training_df[(training_df.ncodpers==15896)]
debug_batch_df7 = training_df[(training_df.ncodpers==15897)]

debug_batch_df_t = pd.concat([debug_batch_df, debug_batch_df1,
                              debug_batch_df2, debug_batch_df3,
                              debug_batch_df4, debug_batch_df5,
                              debug_batch_df6, debug_batch_df7 ])

print debug_batch_df_t.shape
debug_batch_df_t.to_csv("./input/train_batch.csv")
