#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:06:53 2018

@author: jessiechu
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.metrics import classification_report

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
test_label = pd.read_csv('./data/gender_submission.csv')
result = open(r'./result/cluster_report.txt','a+')


print(train.describe())
print(test.describe())

#数据预处理
#由上面可知，Age列的数据是存在缺失值的，因此可以通过中位数来对缺失值进行填补
#对缺失值进行填补,并且将分类数据用0和1来进行代替
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
#下面对Sex这一列来进行处理,其中男性用0表示,女性用1表示
print(train['Sex'].unique())#['male' 'female']
#loc函数表示对标签来进行切片
train.loc[train['Sex'] == 'male', 'Sex'] = 0
train.loc[train['Sex'] == 'female', 'Sex'] = 1
test.loc[test['Sex'] == 'male', 'Sex'] = 0
test.loc[test['Sex'] == 'female', 'Sex'] = 1
#用同样的方法来对Embarked(在何处登船)来进行缺失值处理
print(train['Embarked'].unique())#['S' 'C' 'Q' nan]
print(test['Embarked'].unique())#['S' 'C' 'Q']
#发现这个变量也有缺失值,需要对上面的数据来进行缺失值处理
train['Embarked'] = train['Embarked'].fillna('S')
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'C', 'Embarked'] = 1
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2
test['Embarked'] = test['Embarked'].fillna('S')
test.loc[test['Embarked'] == 'S', 'Embarked'] = 0
test.loc[test['Embarked'] == 'C', 'Embarked'] = 1
test.loc[test['Embarked'] == 'Q', 'Embarked'] = 2
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

test['Survived'] = test_label[['Survived']]

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

clu_kmeans = KMeans(n_clusters=2)
kmeans_pred = clu_kmeans.fit_predict(train[predictors])
print('KMeans:\n',classification_report(kmeans_pred,train['Survived']))
print('KMeans:\n',classification_report(kmeans_pred,train['Survived']), file=result)

clu_mean = MeanShift()
mean_pred = clu_mean.fit_predict(train[predictors])
print('MeanShift:\n',classification_report(mean_pred,train['Survived']))
print('MeanShift:\n',classification_report(mean_pred,train['Survived']),file=result)

def ShowPic(name, cluster):
    t = pd.concat([pd.DataFrame(cluster, columns=['label']), train], axis=1)
    n = len(t['label'].unique())
    colors = ['b','g','r','y','m','k']
    plt.title(name)
    for i in range(n):
        type = t.loc[t['label'] == i]
        plt.scatter(type['Age'], type['Fare'], c=colors[i])
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.savefig('./figure/cluster_result_%s.png' % name)
    plt.show()

ShowPic('KMeans',kmeans_pred)
ShowPic('MeanShift', mean_pred)

