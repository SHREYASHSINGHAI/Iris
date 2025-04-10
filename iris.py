import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


dataset=pd.read_csv(r"D:\college\Iris\iris.csv")

#METHOD 2 FOR GETTING DATASET FROM SKLEARN LIBRARY
# from sklearn import datasets
# iris=datasets.load_iris()
# print(iris)

#SEGREGATING FEATURES AND LABLES
x=dataset[["sepal_length","sepal_width","petal_length","petal_width"]] # features
y=dataset["species"] # labels

#PLOTTING GRAPHS
import seaborn as sns
sns.scatterplot(x="petal_length",y="petal_width",data=dataset,hue="species")
plt.show()


#DEVIDING DATASET FOR TRAINING AND TESTING 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


#APPLYING RANDDOM FOEST
model=RandomForestClassifier()
model.fit(x_train,y_train)
Prediction=model.predict(x_test)


#DATAFRAME FOR COMPARISION
Input=x_test.iloc[:,0]
comparision=pd.DataFrame({"Testing input":Input,"Original value":y_test,"Predicted value":Prediction})
print(comparision)

#GETTING THE ACCURACY OF TH MODEL
print("The accuracy of the model is : ",accuracy_score(y_test,Prediction))