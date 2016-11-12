import pandas as pd
import datetime as dt
import numpy as np
import math
train_df = pd.read_csv('./input/train_batch.csv')
def date_to_span(date_string):
    date_obj = dt.datetime.strptime(date_string, '%Y-%m-%d')
    year_span = 2016 - date_obj.year
    month_span = 6 - date_obj.month
    return year_span * 12 + month_span

indrel_rescale = lambda x: 1 if 1 else 2

# # There is no need to fill this column as it has already been deleted by filtering
# ind_empleado_fill = lambda x: 'P' if np.isnan

sexo_fill = lambda x: 'H' if (x!='H')&(x!='V') else x

antiguedad_rescale = lambda x: -1 if -999999 else x

indrel_1mes_fill = lambda  x: 0 if np.isnan(x) else x

tiprel_1mes_fill = lambda  x: 'I' if (x!='P')&(x!='I')&(x!='A')&(x!='R') else x

conyuemp_fill = lambda x: 'N' if (x!='S')&(x!='N') else x

# canal_entrada_fill = lambda x: 'unknown' if x.isnull() else x

# somehow the empty cell is called 'NA'
cod_prov_fill = lambda x: 0 if 'NA' else x

renta_fill = lambda x: 101490.5 if 'NA' else x

segmento_fill = lambda x: '02 - PARTICULARES' if 'NA' else x

# fill blanks in 'sexo' with 'H'
train_df['sexo'].map(sexo_fill)

# scale -99999 in 'antiguedad' to -1
train_df['antiguedad'].map(antiguedad_rescale)

# # this column is replaced by antiguedad
# convert 'first date in contract' into time span
# train_df['contract_history'] = train_df.fecha_alta.apply(date_to_span)
del train_df['fecha_alta']

# convert '99' in 'indrel' to 2
train_df['indrel'].map(indrel_rescale)

# 'ult_fec_cli_1t', 'indrel_1mes' are deleted
del train_df['ult_fec_cli_1t']
# del train_df['indrel_1mes']

# fill blanks in indrel_1mes
train_df['indrel_1mes'].map(indrel_1mes_fill)

# fill blanks in tiprel_1mes
train_df['tiprel_1mes'].map(tiprel_1mes_fill)

#  fill blanks in 'conyuemp'
train_df['conyuemp'].map(conyuemp_fill)

# # delete this column because there are too many different categories
# train_df['canal_entrada'].map(canal_entrada_fill)
del train_df['canal_entrada']

#  replace 'NA' in 'tipodom' with 0
train_df['cod_prov'].map(cod_prov_fill)

# name of the province is deleted
del train_df['nomprov']

# fill income with median
train_df['renta'].map(renta_fill)

# fill 'segmento' with 02
train_df['segmento'].map(segmento_fill)

train_df.to_csv('./input/filtered_train.csv')
