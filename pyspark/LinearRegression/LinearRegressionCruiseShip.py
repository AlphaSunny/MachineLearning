from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import corr 

spark = SparkSession.builder.appName('cruise').getOrCreate()
df = spark.read.csv('hdfs:///user/maria_dev/MachineLearning/cruise_ship_info.csv',inferSchema=True,header=True)

print("------------------------------------------Cruise line information------------------------------------------")
df.groupBy('Cruise_line').count().show()
indexer = StringIndexer(inputCol="Cruise_line", outputCol="cruise_cat")
indexed = indexer.fit(df).transform(df)


# get the features column
assembler = VectorAssembler(
  inputCols=['Age',
             'Tonnage',
             'passengers',
             'length',
             'cabins',
             'passenger_density',
             'cruise_cat'],
    outputCol="features")

output = assembler.transform(indexed)
output.select("features", "crew").show()
final_data = output.select("features", "crew")
train_data,test_data = final_data.randomSplit([0.7,0.3])

# Create a Linear Regression Model object
lr = LinearRegression(labelCol='crew')

# Fit the model to the data and call this model lrModel
lrModel = lr.fit(train_data)

# Print the coefficients and intercept for linear regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept:  " + str(lrModel.intercept))

test_results = lrModel.evaluate(test_data)
print("RMSE: "+ str(test_results.rootMeanSquaredError))
print("MSE: " + str(test_results.meanSquaredError))
print("R2: " + str(test_results.r2))
# R2 of 0.86 is pretty good, let's check the data a little closer

df.select(corr('crew','passengers')).show()
df.select(corr('crew','cabins')).show()