# -*- coding: utf-8 -*-
"""project1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1puRKJqniv9g-0P1F6BqAidWnqZ8VLy35
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")
#Following these tutorials https://becominghuman.ai/10-machine-learning-projects-to-boost-your-portfolio-88d17e2825b3

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#exploring the data
train['Data'] = 'Train'
test['Data'] = 'Test'
both = pd.concat([train, test], axis=0).reset_index(drop=True)
both['subject'] = '#' + both['subject'].astype(str)

both.head()

#checking basic details and if everything loaded correctly
#missing values should be 0 everywhere. In case it is more, re-load the dataset
def basic_details(df):
    b = pd.DataFrame()
    b['Missing values'] = df.isnull().sum()
    b['N unique values'] = df.nunique()
    b['dtype'] = df.dtypes
    return b
print(basic_details(both))

#plotting the activities
activity = both['Activity']
label_counts = activity.value_counts()

plt.figure(figsize= (16, 8))
plt.bar(label_counts.index, label_counts)

Data = both['Data']
Subject = both['subject']
train = both.copy()
train = train.drop(['Data','subject','Activity'], axis =1)
print(train)

# Standard Scaler
slc = StandardScaler()
train = slc.fit_transform(train)

# dimensionality reduction - experimented with different number of components. The error is the lowest with 0.9
pca = PCA(n_components=0.9, random_state=0)
train = pca.fit_transform(train)
print(train.shape)

X_train, X_test, y_train, y_test = train_test_split(train, activity, test_size = 0.2, random_state = 0)

# Finalizing the model and comparing the test, predict results
results = {}
accuracy = {}

model = KNeighborsClassifier(algorithm='auto', n_neighbors=8, p=1, weights='distance')

_ = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
results["GScv"] = (_.mean(), _.std())

model.fit(X_train, y_train) 
y_predict = model.predict(X_test)

accuracy["GScv"] = accuracy_score(y_test, y_predict)

print(classification_report(y_test, y_predict))

cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)