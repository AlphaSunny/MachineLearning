from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.appName('logregdoc').getOrCreate()

# Load training data
training = spark.read.format("libsvm").load("sample_libsvm_data.txt")

lr = LogisticRegression()

# Fit the model
lrModel = lr.fit(training)

trainingSummary = lrModel.summary

trainingSummary.predictions.show()
# May change soon!
from pyspark.mllib.evaluation import MulticlassMetrics

lrModel.evaluate(training)

# Usually would do this on a separate test set!
predictionAndLabels = lrModel.evaluate(training)

predictionAndLabels.predictions.show()

predictionAndLabels = predictionAndLabels.predictions.select('label','prediction')

predictionAndLabels.show()