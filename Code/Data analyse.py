import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 获取数据
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
#   数据特征提取
train_df.columns.values
#   分析数据特征
train_df.columns.values
# 具体数据，对首尾数据进行查看
train_df.head()

train_df.tail()
# 查看数据结构

train_df.info()
print('-' * 40)
test_df.info()
#   查看数据分布
train_df.describe()
train_df.describe(include=['O'])
# 单个特征与标签的相关性pclass
# pclass、Sex、SibSP和Parch与survive的关系
train_df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex', 'Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby('Parch', as_index=False).mean().sort_values(by='Survived', ascending=False)
# 数据可视化
# 1、年龄与存活的关系
plt.figure(dpi=100)
plt.hist(train_df['Age'], label='All', bins=20)
plt.hist(train_df.loc[train_df.Survived == 1, 'Age'], label='Survived', bins=20)
plt.xlabel('Age')
plt.ylabel('People Number')
plt.legend()
# 2、对上述直方图我们再加入Pclass特征
plt.clf()
fig, axes = plt.subplots(1, 3, dpi=100)
plt.subplots_adjust(left=0, bottom=None, right=2.5, top=None, wspace=None, hspace=None)


def subplot(Pclass, index):
    ax_i = axes[index]
    ax_i.hist(train_df.loc[train_df.Pclass == Pclass, 'Age'], label='All', bins=20)
    ax_i.hist(train_df.loc[(train_df.Survived == 1) & (train_df.Pclass == Pclass), 'Age'], label='Survived', bins=20)
    ax_i.set_xlabel('age')
    ax_i.set_title('Pclass=' + str(Pclass))
    ax_i.set_ylabel('People Number')


for i in range(3):
    subplot(i + 1, i)

plt.legend()

# 3、可视化出发地、性别、社会地位（Pclass）与存活率的关系
grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
# 4、可视化票价与存活率的关系
plt.figure(dpi=100)
plt.hist(train_df['Fare'], label='All', bins=50)
plt.hist(train_df.loc[train_df.Survived == 1, 'Fare'], label='Survived', bins=50)
plt.xlabel('Fare')
plt.ylabel('People Number')
plt.legend()