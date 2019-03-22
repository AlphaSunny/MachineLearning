from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression


spark = SparkSession.builder.appName('lr_example').getOrCreate()

# Load training data
training = spark.read.format("libsvm").load("sample_linear_regression_data.txt")

training.show()