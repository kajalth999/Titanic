import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('train.csv')


dataset.isnull().sum()
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
df=dataset.drop(['Cabin','Name'],axis = 1,inplace = True)
df=dataset.drop(['PassengerId'],axis=1,inplace = True)
df=dataset.drop(['Sex','Parch'],axis=1,inplace=True)
dataset.Age=dataset.Age.astype(int)
cols=['Ticket','Embarked']
le=LabelEncoder()
for col in cols:
    dataset[col]=le.fit_transform(dataset[col])
dataset.Age=dataset.Age.astype(int)
dataset.info()
corr = dataset.corr()
sns.heatmap(corr,annot= True ,cmap = "BuPu")
X=dataset.drop(columns='Survived',axis=1)
y=dataset['Survived']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier =  LogisticRegression(solver='liblinear')
classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500,criterion='entropy',random_state=0,max_depth=10,min_samples_split=2)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,ypred)
cp=confusion_matrix(y_test,y_pred)





