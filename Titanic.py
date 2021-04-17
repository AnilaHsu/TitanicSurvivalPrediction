import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

train = pd.read_csv('Titanic_train.csv')
test = pd.read_csv('Titanic_test.csv')
train['type'] = '1'
test['type']= '2'
all_data = pd.concat([train, test], ignore_index = True)

## 刪除無效列，drop預設刪除行，列需要加axis=1
## inplace = True 會直接修改all_data 的資料，不變數承接
all_data.drop(['PassengerId','Ticket','Name','Cabin'],axis=1,inplace=True)

## Embarked 空值用 C 取代
all_data.Embarked.fillna('C',inplace=True)

## 處理 Fare = 0 或空值, 用平均值取代
all_data.Fare.fillna(0,inplace = True)
fare_df = all_data[all_data['Fare'] != 0]
all_data.loc[all_data['Fare'] == 0 ,'Fare'] = fare_df.Fare.mean()

## 判斷各變數中是否存在缺失值
# print(all_data.isnull().any())
## 各變數中缺失值的數量
# print(all_data.isnull().sum())

all_data = pd.get_dummies(data=all_data,columns=['Sex'])
all_data = pd.get_dummies(data=all_data,columns=['Embarked'])
# print(all_data.dtypes)
org_data = all_data[['type','Survived']]
all_data.drop(['type','Survived'],axis=1,inplace=True)

known_age = all_data[all_data.Age.notnull()]
unknown_age = all_data[all_data.Age.isnull()]

x_train_age = known_age.drop(['Age'], axis=1)
y_train_age = known_age['Age']
x_test_age = unknown_age.drop(['Age'],axis=1)
classifier = KNeighborsClassifier()
classifier.fit(x_train_age,y_train_age.astype('int'))
y_pred_age = classifier.predict(x_test_age)
all_data.loc[all_data.Age.isnull(),'Age'] = y_pred_age

# X = known_age.columns[nomissing.columns != 'Age']
# classifier.fit(known_age[X], known_age.Age)
# y_pred_age = classifier.predict(unknown_age[X])
# all_data.loc[all_data.Age.isnull(),'Age'] = y_pred_age

after_data = pd.concat([all_data,org_data],axis=1)
after_data.loc[after_data['type'] =='1'].to_csv('train_knn.csv')
after_data.loc[after_data['type'] =='2'].to_csv('test_knn.csv')


# 預測 Survived
## 利用train_knn.csv資料,隨機切分30%做測試資料
## 用KNN,SVM,GaussianNB演算法來預測模型的準確度

data = pd.read_csv("train_knn.csv")

feature = data.drop(['type','Survived'], 1)
target = pd.DataFrame(data['Survived'], columns = ['Survived'])
# target = data['Survived']

# print(data)
# print(target)
# print(feature)
x_train, x_test, y_train , y_test = train_test_split(feature, target, test_size=0.3)

# 特徵標準化
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# KNN
classifier = KNeighborsClassifier()
classifier.fit(x_train_std, y_train)
y_test_predicted = classifier.predict(x_test_std)
accuracy = accuracy_score(y_test, y_test_predicted)
print("KNN_Accuracy: ",accuracy)
print(metrics.classification_report(y_test, y_test_predicted))

## GaussianNB
classifier = GaussianNB()
classifier.fit(x_train_std, y_train)
y_test_predicted = classifier.predict(x_test_std)
accuracy = accuracy_score(y_test, y_test_predicted)
print("GaussianNB_Accuracy: ",accuracy)
print(metrics.classification_report(y_test, y_test_predicted))

### SVM
classifier =  SVC()
classifier.fit(x_train_std, y_train)
y_test_predicted = classifier.predict(x_test_std)
accuracy = accuracy_score(y_test, y_test_predicted)
print("Accuracy: ",accuracy)
print(metrics.classification_report(y_test, y_test_predicted))
