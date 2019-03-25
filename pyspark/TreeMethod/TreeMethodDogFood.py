#Tree methods Example
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
spark = SparkSession.builder.appName('dogfood').getOrCreate()

# Load training data
data = spark.read.csv('hdfs:///user/maria_dev/MachineLearning/dog_food.csv',inferSchema=True,header=True)

data.printSchema()

print(data.head())

data.describe().show()



assembler = VectorAssembler(inputCols=['A', 'B', 'C', 'D'],outputCol="features")

output = assembler.transform(data)

from pyspark.ml.classification import RandomForestClassifier,DecisionTreeClassifier

rfc = DecisionTreeClassifier(labelCol='Spoiled',featuresCol='features')

output.printSchema()

final_data = output.select('features','Spoiled')
final_data.head()

rfc_model = rfc.fit(final_data)

print("-----------------feature importance --------------------------------------")
print(rfc_model.featureImportances)

spark.stop()