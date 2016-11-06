import sqlite3 as sql
import pandas as pd
import datetime
import dateutil.relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

product_used_list = []
date_list = []
total_count = 1
normalization = lambda x: x / total_count

connectionPath = "../santander_data.db"
santanderCon = sql.connect(connectionPath)

Saving_Account = "ind_ahor_fin_ult1"
Guarantees = "ind_aval_fin_ult1"
Current_Accounts = "ind_cco_fin_ult1"
Derivada_Account = "ind_cder_fin_ult1"
Payroll_Account = "ind_cno_fin_ult1"
Junior_Account = "ind_ctju_fin_ult1"
Mas_particular_Account = "ind_ctma_fin_ult1"
particular_Account = "ind_ctop_fin_ult1"
particular_Plus_Account = "ind_ctpp_fin_ult1"
Short_term_deposits = "ind_deco_fin_ult1"
Medium_term_deposits = "ind_deme_fin_ult1"
Long_term_deposits = "ind_dela_fin_ult1"
e_account = "ind_ecue_fin_ult1"
Funds = "ind_fond_fin_ult1"
Mortgage = "ind_hip_fin_ult1"
Pensions = "ind_plan_fin_ult1"
Loans = "ind_pres_fin_ult1"
Taxes = "ind_reca_fin_ult1"
Credit_Card = "ind_tjcr_fin_ult1"
Securities = "ind_valo_fin_ult1"
Home_Account = "ind_viv_fin_ult1"
Payroll = "ind_nomina_ult1"
Pensions = "ind_nom_pens_ult1"
Direct_Debit = "ind_recibo_ult1"

start_date = '2015-01-28'
count = 0
current_date = start_date
date_delta = dateutil.relativedelta.relativedelta(months=1)
while(count < 17):
    count = count + 1
    sql_command = "select "+ "sum(" + Saving_Account + "),"\
                  + "sum(" + Guarantees + "),"\
                  + "sum(" + Current_Accounts + "),"\
                  + "sum(" + Derivada_Account + "),"\
                  + "sum(" + Payroll_Account + "),"\
                  + "sum(" + Junior_Account + "),"\
                  + "sum(" + Mas_particular_Account + "),"\
                  + "sum(" + particular_Account + "),"\
                  + "sum(" + particular_Plus_Account + "),"\
                  + "sum(" + Short_term_deposits + "),"\
                  + "sum(" + Medium_term_deposits + "),"\
                  + "sum(" + Long_term_deposits + "),"\
                  + "sum(" + e_account + "),"\
                  + "sum(" + Funds + "),"\
                  + "sum(" + Mortgage + "),"\
                  + "sum(" + Pensions + "),"\
                  + "sum(" + Loans + "),"\
                  + "sum(" + Taxes + "),"\
                  + "sum(" + Credit_Card + "),"\
                  + "sum(" + Securities + "),"\
                  + "sum(" + Home_Account + "),"\
                  + "sum(" + Payroll + "),"\
                  + "sum(" + Pensions + "),"\
                  + "sum(" + Direct_Debit + ")"\
                  + " from santander_train where fecha_dato = '" + current_date + "'"
    product_used_df = pd.read_sql(sql_command, santanderCon)

    count_df = pd.read_sql("select count(*) from santander_train where fecha_dato = '" + current_date + "'", santanderCon)
    total_count = count_df.ix[0,0]
    product_used_df = product_used_df.transpose()
    product_used_list.append(product_used_df[0].map(normalization))

    # first convert date string to datetime format, peform the computation and then convert back
    current_date_dt = datetime.datetime.strptime(current_date, "%Y-%m-%d")
    date_list.append(current_date_dt.date())
    current_date_dt += date_delta
    current_date = datetime.datetime.strftime(current_date_dt, "%Y-%m-%d")

product_time_series = pd.concat(product_used_list, axis=1, keys=date_list)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xlabel("Time")
plt.ylabel("Percentage of Use")
plt.title("Change over time")
plt.gcf().autofmt_xdate()
##****************************************************************##
# uncomment to plot a single product
##****************************************************************##
# plt.plot(date_list, product_time_series.ix['sum(ind_cco_fin_ult1)'])

product_time_series_sum = product_time_series.sum(axis=0)
plt.plot(date_list, product_time_series_sum.values)

plt.show()
# print product_time_series_sum.values
