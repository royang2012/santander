import dataLoader as dl

trainFilePath = './input/train_ver2.csv'
testFilePath = './input/test_ver2.csv'

connectionPath = "./santander_data.db"
chunkSize = 100000
testTableName = "santander_test"
trainTableName = "santander_train"

dl.createTableWithCsvFile(trainFilePath, trainTableName, connectionPath, chunkSize)

sqlStatement1 = "CREATE INDEX index_train_ncodpers ON santander_train (ncodpers)"
sqlStatement2 = ("CREATE INDEX index_train_fechaDato ON santander_train (fecha_dato);")
sqlStatement3 = ("CREATE INDEX index_train_ncodpers_fechaDato ON santander_train (ncodpers, fecha_dato);")

dl.createTableIndex(sqlStatement1, connectionPath)
dl.createTableIndex(sqlStatement2, connectionPath)
dl.createTableIndex(sqlStatement3, connectionPath)

dl.createTableWithCsvFile(testFilePath, testTableName, connectionPath, chunkSize)

sqlStatement1 = "CREATE INDEX index_test_ncodpers ON santander_test (ncodpers)"
sqlStatement2 = ("CREATE INDEX index_test_fechaDato ON santander_test (fecha_dato);")
sqlStatement3 = ("CREATE INDEX index_test_ncodpers_fechaDato ON santander_test (ncodpers, fecha_dato);")

dl.createTableIndex(sqlStatement1, connectionPath)
dl.createTableIndex(sqlStatement2, connectionPath)
dl.createTableIndex(sqlStatement3, connectionPath)
