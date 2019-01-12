#importing the libraries
import xport
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier

#importing the dataset
with open(r'C:\Users\nupur\Desktop\project.xpt', 'rb') as f:
    df1 = xport.to_dataframe(f)

######  DATA Pre-Processing  ######
#Removing columns which has more than 50% null values
df = df1.dropna(axis=1, how='all', thresh=225000, subset=None)

#Removing columns with data having variance less than 0.65 
df_var = df.drop(df.std()[df.std() < 0.65].index.values, axis=1)

#removing columns which will not be required for prediction
df.drop(df.columns[1:16], axis=1, inplace=True)
df.drop(df.select_dtypes(include=['object']), axis=1, inplace= True)

#replacing the naN values of the dataset with mode of that columns 
for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)
print(df.isna().sum())

#Extract the label from the dataset 
label = df['_BMI5CAT']
label = df['_BMI5CAT'].apply({1:'Not Obese', 2:'Not Obese', 3:'Obese', 4:'Obese'}.get)

df['Label']= label
df.drop(columns=['_BMI5CAT'], axis=1, inplace=True)

######  Splitting into training and test datset  ######

sample = np.random.rand (len (df)) < 0.8
training = df[sample]
testing = df[~sample]
train_data = training.values
test_data = testing.values
training_attributes = training.iloc[:,0:167]
training_label = training.iloc[:,-1]
testing_attributes = testing.iloc[:,0:167]
testing_label = testing.iloc[:,-1]

######  Implementation of Decision Tree  ######
decision_tree = DecisionTreeClassifier(criterion = "entropy", max_depth = 1, max_features = 1, random_state = 1)
print(decision_tree)
decision_tree.fit(training_attributes,training_label)
training_prediction = decision_tree.predict(training_attributes)
testing_prediction = decision_tree.predict(testing_attributes)
sum(training_prediction != training_label)/360075
sum(testing_prediction != testing_label)/89941

######  Implementation of Adaboost  ######
#initializing the weight, w= 1/number of samples
weight = (np.ones(len(training_attributes)) / len(training_attributes)) 
ada_training = np.zeros(len(training_attributes))
ada_testing = np.zeros(len(training_attributes))
#weight = 1/len(training_attributes)
#print(weight)
print(weight)
alpha1 = []
hypothesis1 = []
#Performing two iterations
for i in range(2): 
    #Using decision tree classifier as the hypothesis
    decision_tree.fit(training_attributes, training_label, sample_weight = weight)
    pred_training_i = decision_tree.predict(training_attributes)
    pred_testing_i = decision_tree.predict(testing_attributes)
    #checking whether the predicted value is equal to the original label
    hypothesis = [int(x) for x in (pred_training_i != training_label)]
    updated_hypothesis = [x if x==1 else -1 for x in hypothesis]
    error_t = np.dot(weight,hypothesis) / sum(weight)
    #calculating the value of alpha
    alpha = 0.5 * np.log( (1 - error_t) / float(error_t))
    alpha1.append(alpha)
    hypothesis1.append(hypothesis)
    weight = np.multiply(weight, np.exp([float(x) * alpha for x in updated_hypothesis]))
H = np.multiply(int(alpha1[0]),hypothesis1[0]) + np.multiply(int(alpha1[1]),hypothesis1[1])
print("Strong classifier: %s" %H)
ada_training = [sum(x) for x in zip(ada_training, [x * alpha for x in pred_training_i])]
ada_testing = [sum(x) for x in zip(ada_testing, [x * alpha for x in pred_testing_i])]
ada_training = np.sign(ada_training)
ada_testing = np.sign(ada_testing)
error_boosting_train = sum(ada_training != training_label)/360075
error_boosting_test = sum(ada_testing != testing_label)/89941
print(error_boosting_train)
print(error_boosting_test)




