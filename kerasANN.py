import tensorflow as tf
import keras

import pandas as pd
# Read the data into a Pandas DataFrame
df = pd.read_csv('heart_disease_uci.csv')
# Display the first few rows of the DataFrame
print(df.head())
# Check for duplicates
print(df[df.duplicated()])
# Drop duplicates(if any)
#df = df.drop_duplicates()
# Check for number of categories
print(df.nunique())
#check for information
print(df.describe())
# Check for missing values
print(df.isnull().sum())
# Handling missing values 
# df = df.fillna('NA')
df = df.drop(columns=['id','ca','thal','slope'])
df["trestbps"].fillna(df["trestbps"].mean(), inplace = True)
df["chol"].fillna(df["chol"].mean(), inplace = True)
df["thalch"].fillna(df["thalch"].mean(), inplace = True)
df["oldpeak"].fillna(df["oldpeak"].mean(), inplace = True)
df["exang"].fillna(df["exang"].mode()[0], inplace = True)
df["fbs"].fillna(df["fbs"].mode()[0], inplace = True)
df["restecg"].fillna(df["restecg"].mode()[0], inplace = True)
print(df.isnull().sum())
print(df.info())
print(df.nunique())


import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(data=df,orient='h')
# Show the plot
plt.title('Boxplot of Value')
plt.show()
  
columns_to_check = ['chol', 'trestbps','thalch']
# Loop through columns and identify outliers
for column in columns_to_check:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
sns.boxplot(data=df,orient='h')
# Show the plot
plt.title('Boxplot of Value')
plt.show()

print(df.nunique())
print(df['restecg'].value_counts())

#apply onehotencoder to object datatype
object_col=df.select_dtypes(include=["object"]).columns
df=pd.get_dummies(df,columns=object_col)
df=df.replace({True:1,False:0})
print(df.info())
#!!!!!!!!!!!======>

#Import Libraries
import numpy as np
#read data and make preprocessing we do it before
#now divide data
x=df.iloc[ : , 0:-1 ]
y=df.iloc[ : ,-1 ]

#Data preprocessing
# didvde data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3, random_state=0,stratify=y)
#we perform feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Build and visualize the Artificial Neural Network
ann = tf.keras.models.Sequential()
# Add the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=12, activation='relu', input_shape=X_train[0].shape))
# Add the second hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))
# Add the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Training the ANN
#Evaluating the model







