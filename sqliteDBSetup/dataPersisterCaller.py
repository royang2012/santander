import dataPersister as dp

# trainFilePath = '../input/train_ver2.csv'
# testFilePath = '../input/test_ver2.csv'

connectionPath = "../santander_data.db"
chunkSize = 100000
testTableName = "santander_test"
trainTableName = "santander_train"

# dp.createTableWithCsvFile(trainFilePath, trainTableName, connectionPath, chunkSize)
#
# sqlStatement1 = "CREATE INDEX index_train_ncodpers ON santander_train (ncodpers)"
# sqlStatement2 = ("CREATE INDEX index_train_fechaDato ON santander_train (fecha_dato);")
# sqlStatement3 = ("CREATE INDEX index_train_ncodpers_fechaDato ON santander_train (ncodpers, fecha_dato);")
# sqlStatement4 = ("CREATE INDEX index_train_age_ncodpers ON santander_train (age, ncodpers);")
#
# dp.createTableIndex(sqlStatement1, connectionPath)
# dp.createTableIndex(sqlStatement2, connectionPath)
# dp.createTableIndex(sqlStatement3, connectionPath)
# dp.createTableIndex(sqlStatement4, connectionPath)
#
# dp.createTableWithCsvFile(testFilePath, testTableName, connectionPath, chunkSize)
#
# sqlStatement1 = "CREATE INDEX index_test_ncodpers ON santander_test (ncodpers)"
# sqlStatement2 = ("CREATE INDEX index_test_fechaDato ON santander_test (fecha_dato);")
# sqlStatement3 = ("CREATE INDEX index_test_ncodpers_fechaDato ON santander_test (ncodpers, fecha_dato);")
#
# dp.createTableIndex(sqlStatement1, connectionPath)
# dp.createTableIndex(sqlStatement2, connectionPath)
# dp.createTableIndex(sqlStatement3, connectionPath)

# index_statement1 = "create index index_ncodpers ON added1605 (ncodpers)"
# dp.createTableIndex(index_statement1, connectionPath)

index_statement2 = "create index index_ncodpers ON santander_train7 (ncodpers)"
index_statement3 = "create index index_fecha_dato ON santander_train7 (fecha_dato)"
index_statement4 = "create index index_ncodpers_fecha ON santander_train7 (ncodpers, fecha_dato)"
dp.createTableIndex(index_statement2, connectionPath)
# dp.createTableIndex(index_statement3, connectionPath)
# dp.createTableIndex(index_statement4, connectionPath)
