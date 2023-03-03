import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ��ȡ����
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
#   ����������ȡ
train_df.columns.values
#   ������������
train_df.columns.values
# �������ݣ�����β���ݽ��в鿴
train_df.head()

train_df.tail()
# �鿴���ݽṹ

train_df.info()
print('-' * 40)
test_df.info()
#   �鿴���ݷֲ�
train_df.describe()
train_df.describe(include=['O'])
# �����������ǩ�������pclass
# pclass��Sex��SibSP��Parch��survive�Ĺ�ϵ
train_df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex', 'Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby('Parch', as_index=False).mean().sort_values(by='Survived', ascending=False)
# ���ݿ��ӻ�
# 1����������Ĺ�ϵ
plt.figure(dpi=100)
plt.hist(train_df['Age'], label='All', bins=20)
plt.hist(train_df.loc[train_df.Survived == 1, 'Age'], label='Survived', bins=20)
plt.xlabel('Age')
plt.ylabel('People Number')
plt.legend()
# 2��������ֱ��ͼ�����ټ���Pclass����
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

# 3�����ӻ������ء��Ա�����λ��Pclass�������ʵĹ�ϵ
grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
# 4�����ӻ�Ʊ�������ʵĹ�ϵ
plt.figure(dpi=100)
plt.hist(train_df['Fare'], label='All', bins=50)
plt.hist(train_df.loc[train_df.Survived == 1, 'Fare'], label='Survived', bins=50)
plt.xlabel('Fare')
plt.ylabel('People Number')
plt.legend()