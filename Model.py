from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('awsproject').getOrCreate()
data = spark.read.csv('credit_data.csv',inferSchema=True, header=True)
data.printSchema()
data.describe().show()
data.columns
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['income','age','loan', 'LTI'],outputCol='features')
output = assembler.transform(data)
final_data = output.select('features','default')
train_default,test_default = final_data.randomSplit([0.67,0.33])
from pyspark.ml.classification import LogisticRegression
lr_default = LogisticRegression(labelCol='default')
fitted_default_model = lr_default.fit(train_default)
training_sum = fitted_default_model.summary
training_sum.predictions.describe().show()
from pyspark.ml.evaluation import BinaryClassificationEvaluator
pred_and_labels = fitted_default_model.evaluate(test_default)
pred_and_labels.predictions.show()
new_customers = spark.read.csv('new_customers.csv',inferSchema=True,header=True)
final_lr_model = lr_default.fit(final_data)
test_new_customers = assembler.transform(new_customers)
final_results = final_lr_model.transform(test_new_customers)
final_results.select('clientid','prediction').show()
