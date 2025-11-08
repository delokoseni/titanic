import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix  # type: ignore

df = pd.read_csv('titanic.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Embarked'].fillna('S', inplace=True)
age_1 = df[df['Pclass'] == 1]['Age'].median()
age_2 = df[df['Pclass'] == 2]['Age'].median()
age_3 = df[df['Pclass'] == 3]['Age'].median()

def fill_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return age_1
        elif row['Pclass'] == 2:
            return age_2
        elif row['Pclass'] == 3:
            return age_3
    return row['Age']

df['Age'] = df.apply(fill_age, axis = 1)

def fill_sex(sex):
    if sex == 'male':
        return 1
    return 0

df['Sex'] = df['Sex'].apply(fill_sex)
df[list(pd.get_dummies(df['Embarked']).columns)] = pd.get_dummies(df['Embarked'])
df.drop('Embarked', axis=1, inplace=True)

def is_alone(row):
    if (row['SibSp'] + row['Parch']) == 0:
        return 1
    return 0

df['Alone'] = df.apply(is_alone, axis = 1)
x = df.drop('Survived', axis = 1)
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
percent = accuracy_score(y_test, y_pred) * 100
confusion_matrix = confusion_matrix(y_test, y_pred)
print(percent)
print(confusion_matrix)



