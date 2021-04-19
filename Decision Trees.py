#IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

#Change Working Directory if required
import os
os.getcwd()
os.chdir('C:\\Users\\User\\Desktop\\school\\Python\\projects\\DT and RFs')

#DATA IMPORT
loans = pd.read_csv('C:\\Users\\User\\Documents\\loan_data.csv')

#EDA
print('\n',loans.info(),'\n')
print('\n',loans.describe(),'\n')
print('\n',loans.head(),'\n')

#pandas vis
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].plot.hist(bins=35,color='blue',alpha=0.5,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].plot.hist(bins=35,color='red',alpha=0.5,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.tight_layout()
plt.savefig('FICO Histogram.jpg')
plt.show()

#Seaborn vis
i1 = sns.displot(data=loans,x='fico',bins=35,aspect=2,hue='credit.policy',palette='Set1')
plt.tight_layout()
i1.savefig('seaborn FICO.jpg')
plt.show()

#Not Fully paid vis
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].plot.hist(bins=35,color='blue',alpha=0.5,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].plot.hist(bins=35,color='red',alpha=0.5,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')
plt.tight_layout()
plt.savefig('Not Fully Paid.jpg')
plt.show()

#loans by purpose
plt.figure(figsize=(10,6))
sns.countplot(data=loans,x='purpose',hue='not.fully.paid',palette='Set1')
plt.tight_layout()
plt.savefig('loans by purpose.jpg')
plt.show()

#FICO vs Interest Rate
i2 = sns.jointplot(data=loans,x='fico',y='int.rate',color='purple')
plt.tight_layout()
i2.savefig('FICO vs Interest Rate.jpg')
plt.show()

#Regression Plot
i3 = sns.lmplot(data=loans,x='fico',y='int.rate',col='not.fully.paid',hue='credit.policy',palette='Set1')
plt.tight_layout()
i3.savefig('Fico vs Interest Rate regression.jpg')
plt.show()

#DATA CLEANING

#Dealing with categorical variables
final_loans = pd.get_dummies(data=loans,columns=['purpose'],drop_first=True)
print('\n',final_loans.head(),'\n')

#Train Test Split
from sklearn.model_selection import train_test_split

X = final_loans.drop('not.fully.paid',axis=1)
y = final_loans['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

#Metrics
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix

print('Dtree Classification Report:','\n',classification_report(y_test,predictions),'\n')
print('Dtree Confusion Matrix:''\n',confusion_matrix(y_test,predictions),'\n')

sns.set_style('white')
plot_confusion_matrix(dtree,X_test,y_test)
plt.savefig('Dtree Confusion Matrix.jpg')
plt.show()

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500)

rfc.fit(X_train,y_train)
preds = rfc.predict(X_test)

#metrics
print('RFC Classification Report:','\n',classification_report(y_test,preds),'\n')
print('RFC Confusion Matrix:','\n',confusion_matrix(y_test,preds),'\n')

plot_confusion_matrix(rfc,X_test,y_test)
plt.savefig('RFC Confusion Matrix.jpg')
plt.show()

print('Depends what metric you are trying to optimize for.\n Notice the recall for each class for the models.\nNeither did very well.\n')

