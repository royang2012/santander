import sqlite3 as sql
import pandas as pd
# import pandas.io.sql as pd_sql
#
# import datetime as dt
#
# import matplotlib.pyplot as plt

connectionPath = "./santander_data.db"
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

train_total_count = pd.read_sql("select count(*) from santander_train", santanderCon)

product_used_df = pd.read_sql("select "
                              + "sum(" + Saving_Account + "),"
                              + "sum(" + Guarantees + "),"
                              + "sum(" + Derivada_Account + "),"
                              + "sum(" + Payroll_Account + "),"
                              + "sum(" + Junior_Account + "),"
                              + "sum(" + Mas_particular_Account + "),"
                              + "sum(" + particular_Account + "),"
                              + "sum(" + particular_Plus_Account + "),"
                              + "sum(" + Short_term_deposits + "),"
                              + "sum(" + Medium_term_deposits + "),"
                              + "sum(" + Long_term_deposits + "),"
                              + "sum(" + e_account + "),"
                              + "sum(" + Funds + "),"
                              + "sum(" + Mortgage + "),"
                              + "sum(" + Pensions + "),"
                              + "sum(" + Loans + "),"
                              + "sum(" + Taxes + "),"
                              + "sum(" + Credit_Card + "),"
                              + "sum(" + Securities + "),"
                              + "sum(" + Home_Account + "),"
                              + "sum(" + Payroll + "),"
                              + "sum(" + Pensions + "),"
                              + "sum(" + Direct_Debit + ")"
                              + " from santander_train where ind_cco_fin_ult1 = 1 ", santanderCon)
print product_used_df.shape
transposed_df = product_used_df.transpose()
sorted_df = transposed_df.sort_values(by = 0, axis = 0, ascending = False)
sorted_df = sorted_df / 13647309
print sorted_df / 0.65
# count_by_age["age"] = pd.to_numeric(count_by_age["age"], errors="coerce")
#
# width = 1
# plt.bar(count_by_age["age"], count_by_age["count"], width, color="tomato")
# plt.xlabel("Age")
# plt.ylabel("Count")
# plt.title("Age Distribution")
# plt.show()
