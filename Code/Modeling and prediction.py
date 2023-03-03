import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 获取数据
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')

# 1、删除无效特征-删除Passengerld【可进行保存】、Ticket、Cabin特征
data_id = []
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
data_id.append(train_df.pop('PassengerId'))
data_id.append(test_df.pop('PassengerId'))

train_df.shape, test_df.shape

# ((891, 9), (418, 8))
# 2、提取有效的特征

train_df['Title'] = train_df.Name.str.extract('([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
# 发现某些类别特别少，所占比例不均匀，且“Mlle”、“Ms”都是“Miss”，“Mme”应该为“Mrs”，可以对其进行替换，对于数量少的可以进行整合为一个类别“Rare”。

combine = [train_df, test_df]
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# 再对所有的类别进行映射：
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
# 删除Name特征：
train_df = train_df.drop('Name', axis=1)
test_df = test_df.drop('Name', axis=1)
combine = [train_df, test_df]
# 3、转换特征
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
# 4、缺失值填充
# Age填充
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            age_mean = guess_df.mean()
            age_std = guess_df.std()
            age_guess = np.random.uniform(age_mean - age_std, age_mean + age_std)

            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = age_guess

    dataset['Age'] = dataset['Age'].astype(int)
train_df['AgeBand'] = pd.cut(train_df['Age'], 10)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
train_df['Age'] = pd.cut(train_df['Age'], bins=10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test_df['Age'] = pd.cut(test_df['Age'], bins=10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
train_df['Age'] = train_df['Age'].astype('int64')
test_df['Age'] = test_df['Age'].astype('int64')
combine = [train_df, test_df]
# Embarked填充
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset.Embarked.dropna().mode()[0])
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# Fare
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['Fare'] = pd.qcut(train_df['Fare'], 3, labels=[0, 1, 2])
test_df['Fare'] = pd.qcut(test_df['Fare'], 3, labels=[0, 1, 2])
train_df['Fare'] = train_df['Fare'].astype('int64')
test_df['Fare'] = test_df['Fare'].astype('int64')
combine = [train_df, test_df]
# 5、特征组合（特征工程）将SibSp与Parch组合，得到Family特征
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# 6、训练数据
# 导入数据
X_train = np.array(train_df.drop("Survived", axis=1))
Y_train = np.array(train_df["Survived"])
X_test  = np.array(test_df)
X_train.shape, Y_train.shape, X_test.shape
# 模型训练，采用五折交叉验证：
def train(clf):
    kf = KFold(n_splits=5)
    score = list()
    for train_index, test_index in kf.split(range(len(X_train))):
        train_X, test_X = X_train[train_index], X_train[test_index]
        train_y, test_y = Y_train[train_index], Y_train[test_index]
        clf.fit(train_X, train_y)
        score.append(clf.score(test_X, test_y))
    avg_score = np.mean(score)
    print(avg_score)
    return avg_score
# Logistic 回归
clf = LogisticRegression()
lr = train(clf)
# 决策树
clf = DecisionTreeClassifier()
cart = train(clf)
# KNN
clf = KNeighborsClassifier(n_neighbors=3)
knn = train(clf)
# 朴素贝叶斯
clf = GaussianNB()
nb = train(clf)
# SVM
clf = SVC(kernel='rbf')
svm = train(clf)
# MLP
clf = MLPClassifier(max_iter=1000, learning_rate='adaptive')
mlp = train(clf)

#提交结果
clf = SVC(kernel='rbf')
svm = train(clf)
pred = clf.predict(X_test)
pred = np.c_[data_id[1], pred]
pred_df = pd.DataFrame(pred, columns=['PassengerId', 'Survived'])
pred_df.to_csv('/kaggle/working/Submission.csv', index=False)
