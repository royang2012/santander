import pandas as pd

import pandas as pd
from sqlalchemy import create_engine # database connection
import datetime as dt
from IPython.display import display

import plotly.plotly as py # interactive graphing
from plotly.graph_objs import Bar, Scatter, Marker, Layout

# display(pd.read_csv('./input/train_ver2.csv', nrows=2).head())
trainFilePath = './input/train_ver2.csv'

disk_engine = create_engine('sqlite:///santander_data.db')

start = dt.datetime.now()
chunksize = 50000
j = 0
index_start = 1

for train_df in pd.read_csv(trainFilePath, chunksize=chunksize, iterator=True, encoding='utf-8'):

    train_df['fecha_dato'] = pd.to_datetime(train_df['fecha_dato']) # Convert to datetimes

    train_df.index += index_start

    j+=1
    print '{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j*chunksize)

    train_df.to_sql('santander_train', disk_engine, if_exists='append')
    index_start = train_df.index[-1] + 1

df = pd.read_sql_query('SELECT * FROM santander_train LIMIT 3', disk_engine)
df.head()
