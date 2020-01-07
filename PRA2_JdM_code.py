# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es el archivo PRA2_JdM_code

Created on Fri Jan  3 18:50:58 2020

@author: Jordi de Mas
"""

# Importamos las librerias Python que usaremos
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Cargamos los datos que hemos obtenido de la web de Kaggle
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Veamos algunos registros del fichero train
df_train.head()

# Comprobamos los tipos de datos y la existencia de "missing values"
df_train.info()

# Eliminamos las columnas que, a priori, carecen de interés para el análisis.
df_cleaned = df_train.loc[:, ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

# Hacemos lo mismo para el fichero de test.
df_test = df_test.loc[:, ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

# Vemos como queda el conjunto de datos después de eliminar las columnas indicadas.
df_cleaned.head()

# Sustituimos los missing values del campo Embarked (sólo 2) por el mayoritariamente presente en el fichero.
df_cleaned['Embarked'].describe()

# Sustituimos los missing values de Embarked con el valor "S" que es el mayoritario.
df_cleaned['Embarked'] = df_cleaned['Embarked'].fillna('S')
# Seguimos el mismo criterio en el fichero de test.
df_test['Embarked'] = df_test['Embarked'].fillna('S')

# Sustituimos los missing values de Age por la mediana de las edades.
age_median = df_cleaned['Age'].median()
df_cleaned['Age'] = df_cleaned['Age'].fillna(age_median)
# Hacemos lo mismo para el fichero de test
age_median_t = df_test['Age'].median()
df_test['Age'] = df_test['Age'].fillna(age_median_t)

# Cambiamos la variable Sex por valores numéricos
df_cleaned['Sex'].replace(['male', 'female'], [0, 1], inplace = True)
# Lo cambiamos también en el fichero de test
df_test['Sex'].replace(['male', 'female'], [0, 1], inplace = True)

# Cambiamos la variable Age por valores numéricos (int64)
age2 = df_cleaned['Age'].copy()
df_cleaned['Age'] = age2.astype(int)
# También lo cambiamos en el fichero de test
age3 = df_test['Age'].copy()
df_test['Age'] = age3.astype(int)

# Cambiamos también los datos del puerto de embarque por números
df_cleaned['Embarked'].replace(['C', 'Q', 'S'], [0, 1, 2], inplace = True)
# Hacemos lo mismo en el fichero de test
df_test['Embarked'].replace(['C', 'Q', 'S'], [0, 1, 2], inplace = True)

# Vemos como queda el fichero después de los cambios
df_cleaned.info()
df_cleaned.head()

# Identificación y tratamiento de outliers
f = plt.figure(figsize = (15, 5));
plt.subplot(1,4,1)
sns.boxplot(x = df_cleaned['Survived'])
plt.subplot(1,4,2)
sns.boxplot(x = df_cleaned['Pclass'])
plt.subplot(1,4,3)
sns.boxplot(x = df_cleaned['Sex'])
plt.subplot(1,4,4)
sns.boxplot(x = df_cleaned['Age'])

f = plt.figure(figsize = (15, 5));
plt.subplot(1,3,1)
sns.boxplot(x = df_cleaned['SibSp'])
plt.subplot(1,3,2)
sns.boxplot(x = df_cleaned['Parch'])
plt.subplot(1,3,3)
sns.boxplot(x = df_cleaned['Embarked'])

# Creamos intervalos de edades.
#bins = [0, 10, 18, 25, 40, 60, 100]
#names = ['1', '2', '3', '4', '5', '6']
#df_cleaned['Age'] = pd.cut(df_cleaned['Age'], bins, labels = names)
#df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)

# Comprobamos la nueva distribución de edades
#df_cleaned['Age'].value_counts()

# Analizamos la distribución de salvados por sexo.
df_cleaned.groupby(['Survived','Sex']).count().PassengerId
df_cleaned.groupby(['Survived','Sex']).count().PassengerId.plot(kind='bar')

# Distribución de salvados por clase.
df_cleaned.groupby(['Survived','Pclass']).count().PassengerId
df_cleaned.groupby(['Survived','Pclass']).count().PassengerId.plot(kind='bar')

# Test de normalidad de Anderson-Darling
print('Test Anderson-Darling - Pclass: ')
result_anderson = scipy.stats.anderson(df_cleaned['Pclass'], dist = 'norm')
for i in range(len(result_anderson.critical_values)):
    sl, cv = result_anderson.significance_level[i], result_anderson.critical_values[i]
    if result_anderson.statistic < result_anderson.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

print('')
print('Test Anderson-Darling - Sex: ')
result_anderson = scipy.stats.anderson(df_cleaned['Sex'], dist = 'norm')
for i in range(len(result_anderson.critical_values)):
    sl, cv = result_anderson.significance_level[i], result_anderson.critical_values[i]
    if result_anderson.statistic < result_anderson.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

print('')
print('Test Anderson-Darling - SibSp: ')
result_anderson = scipy.stats.anderson(df_cleaned['SibSp'], dist = 'norm')
for i in range(len(result_anderson.critical_values)):
    sl, cv = result_anderson.significance_level[i], result_anderson.critical_values[i]
    if result_anderson.statistic < result_anderson.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

print('')
print('Test Anderson-Darling - Parch: ')
result_anderson = scipy.stats.anderson(df_cleaned['Parch'], dist = 'norm')
for i in range(len(result_anderson.critical_values)):
    sl, cv = result_anderson.significance_level[i], result_anderson.critical_values[i]
    if result_anderson.statistic < result_anderson.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

print('')
print('Test Anderson-Darling - Embarked: ')
result_anderson = scipy.stats.anderson(df_cleaned['Embarked'], dist = 'norm')
for i in range(len(result_anderson.critical_values)):
    sl, cv = result_anderson.significance_level[i], result_anderson.critical_values[i]
    if result_anderson.statistic < result_anderson.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

# Homogeneidad de varianzas mediante test de Fligner-Killeen
class1 = df_cleaned[df_cleaned['Pclass'] == '1']
class2 = df_cleaned[df_cleaned['Pclass'] == '2']
class3 = df_cleaned[df_cleaned['Pclass'] == '3']
stat1, pval1 = scipy.stats.fligner(class1, class2, class3, center = 'median')
stat2, pval2 = scipy.stats.fligner(class1, class2, class3, center = 'trimmed')
print(stat1, pval1)
print(stat2, pval2)

# Empezamos a construir modelos de Machine Learning
# Separamos la columna de Survived del fichero de entrenamiento.
X_train = df_cleaned.drop('Survived', axis = 1)
Y_train = df_cleaned['Survived']
X_test = df_test

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# Linear Support Vector Machine
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

# Análisis de Random Forest usando K-Fold Cross Validation
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# Análisis de Decision Tree usando K-Fold Cross Validation
dt = DecisionTreeClassifier()
scores = cross_val_score(dt, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# Importancia de las variables en Random Forest.
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)

# Graficamos los resultados
importances.plot.bar()

# Matriz de confusión
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=10)
confusion_matrix(Y_train, predictions)

# Precision and Recall
print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))

# Calculamos ROC AUC curve (true positive rate and false positive rate). Se dibujan uno contra otro.
# Calculamos las probabilidades de la predicción
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1.05], [0, 1.05], 'r', linewidth=4)
    plt.axis([0, 1.05, 0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()

# Calculamos el ROC AUC Score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)
