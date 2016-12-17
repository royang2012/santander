import pandas as pd
prov_df = pd.read_csv('../input/prov_info.csv')

# mapping province to number of santander offices
for i in range(0, prov_df.shape[0]):
    p_s = 'dict_prov_off[' +'"'+prov_df.ix[i, 'alias']+'"]= '
    print p_s, prov_df.ix[i, 'office'], ';'

# mappping province to office per area
for i in range(0, prov_df.shape[0]):
    p_s = 'dict_prov_opa[' +'"'+prov_df.ix[i, 'alias']+'"]= '
    print p_s, prov_df.ix[i, 'officePerArea'], ';'

# mappping province to office per area
for i in range(0, prov_df.shape[0]):
    p_s = 'dict_prov_gdp[' +'"'+prov_df.ix[i, 'alias']+'"]= '
    print p_s, prov_df.ix[i, 'GDPPerC'], ';'

# mappping province to office per population
for i in range(0, prov_df.shape[0]):
    p_s = 'dict_prov_opp[' +'"'+prov_df.ix[i, 'alias']+'"]= '
    print p_s, prov_df.ix[i, 'officePerP'], ';'
