#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 20:24:12 2018

@author: jessiechu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
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
print(test['Embarked'].unique())#['S' 'C' 'Q' nan]
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

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

rfc = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
rfc.fit(train[predictors], train['Survived'])
rf = rfc.predict(test[predictors])


decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(train[predictors], train['Survived'])
dt = decision_tree_classifier.predict(test[predictors])
decision_tree_classifier = DecisionTreeClassifier(max_depth=4, max_features=5)
#print(cv_scores)

def ShowPic(name, test_Y, i):
    t = pd.concat([pd.DataFrame(test_Y, columns=['Survived']), test[['Age', 'Fare']]], axis=1)
    alive = t.loc[t['Survived'] == 1]
    dead = t.loc[t['Survived'] == 0]
    figure1 = plt.subplot(2, 1, i)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    figure1.set_title(name)
    alive_distribute = figure1.scatter(alive['Age'], alive['Fare'], c='green', marker='o')
    dead_distribute = figure1.scatter(dead['Age'], dead['Fare'], c='black', marker='x')
    plt.xlabel('Age')
    plt.ylabel('Fare')
    figure1.legend((alive_distribute, dead_distribute), ('survived', 'dead'), loc=1)
#    plt.savefig()


plt.figure()
ShowPic('RandomForest',rf, 1)
ShowPic('DecisionTree', dt, 2)
plt.show()


