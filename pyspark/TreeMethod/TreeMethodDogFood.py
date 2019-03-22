#Tree methods Example
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('dogfood').getOrCreate()

# Load training data
data = spark.read.csv('dog_food.csv',inferSchema=True,header=True)

data.printSchema()

data.head()

data.describe().show()

# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=['A', 'B', 'C', 'D'],outputCol="features")

output = assembler.transform(data)

from pyspark.ml.classification import RandomForestClassifier,DecisionTreeClassifier

rfc = DecisionTreeClassifier(labelCol='Spoiled',featuresCol='features')

output.printSchema()

final_data = output.select('features','Spoiled')
final_data.head()

rfc_model = rfc.fit(final_data)

rfc_model.featureImportances