from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
spark = SparkSession.builder.appName('cluster').getOrCreate()

# Loads data.
dataset = spark.read.format("libsvm").load("hdfs:///user/maria_dev/MachineLearning/sample_kmeans_data.txt")

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)