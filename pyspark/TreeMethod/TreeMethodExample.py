from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('rf').getOrCreate()


# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

data.show()

data.head()

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

trainingData.printSchema()

# Train model.  This also runs the indexers.
model = rf.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

predictions.printSchema()

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

# Not a very good example to show this!
model.featureImportances

from pyspark.ml.classification import GBTClassifier

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("sample_libsvm_data.txt")


# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a GBT model.
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

# Train model.  This also runs the indexers.
model = gbt.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))