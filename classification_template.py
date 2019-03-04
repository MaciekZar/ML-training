"""
Classification - set of tools

"""
# basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('    ')
X = df.iloc[:, :].values
y = df.iloc[:, :].values

# Splitting the dataset: training set,  test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# scaling
from sklearn.preprocessing import StandardScaler
stsc = StandardScaler()
X_train = stsc.fit_transform(X_train)
X_test = stsc.transform(X_test)


#Basic table with results
df3 = pd.DataFrame({'y_pred':y_pred, 'y_result':y_test})
df3['difference'] = df3['y_result'] == df3['y_pred']
print(df3)
print(df3['difference'].value_counts())
# percentage of bad predictions
print(df3['difference'].value_counts()[0]/df3['difference'].value_counts()[1])


# Confusion Matrix
from sklearn.metrics import confusion_matrix
Cm = confusion_matrix(y_test, y_pred)
print(Cm)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#accuracy rate
accR = (tn+tp)/(tn+fp+fn+tp)
print(accR)

#classification report
from sklearn.metrics import classification_report
tst = classification_report(y_test, y_pred)
print(tst)


# Visualising the Test set  (by www.superdatascience.com)
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#error_rate test
error_rate = []
#for RandomForest more is needed, ex. 1,500
for i in range(1,40):
    #RandomForestClassifier(n_estimators = i, criterion = 'entropy', random_state = 0)
    knn = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p = 2)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


#error rate test visualisation:
plt.figure(figsize=(10,6))
#range for RandomForest must be changed
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


#Cumulative accuracy profile
from sklearn.metrics import auc
from scipy.integrate import simps

def cap(y_test, y_pred, plot=True, rule = 'trapezoid'):
    """Cumulative accuracy profile - plot and/or Accurancy rating (AR)"""
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    df.sort_values(by=['y_pred'], ascending=False, inplace=True)
    sUM = sum(y_test)
    #deafault = show plots
    if plot == True:
        plt.plot(range(len(y_pred)), np.cumsum(df['y_test']) / sUM, label='actual model')
        plt.plot(range(len(y_pred)), np.cumsum([1 if n < sUM else 0 for n, x in enumerate(y_test)]) / sUM,
                 label='perfect model')
        plt.plot(range(len(y_pred)), np.cumsum([sUM / len(y_pred) for y in range(len(y_pred))]) / sUM, label='random model')
        plt.ylabel('% of positive outcome')
        plt.xlabel('% of data')
        plt.title('Cumulative accuracy profile')
        plt.legend()
        plt.show()
    #Accurancy Rate - deafult: trapezoid rule
    # Trapezoidal rule
    if rule == 'trapezoid':
        #area below actual model curve
        A = auc(range(len(y_pred)), np.cumsum(df['y_test']))
        #area below perfect model curve
        B = (auc(range(len(y_pred)), np.cumsum([1 if n < sUM else 0 for n, x in enumerate(y_test)])))
        #area below random model curve
        C = auc(range(len(y_pred)), np.cumsum([sUM/len(y_pred) for y in range(len(y_pred))]))
    #Simpson's rule
    elif rule == 'simpson':
        #area below actual model curve
        A = simps(np.cumsum(df['y_test']))
        #area below perfect model curve
        B = simps(np.cumsum([1 if n < sUM else 0 for n, x in enumerate(y_test)]))
        #area below random model curve
        C = simps([sUM/len(y_pred) for y in range(len(y_pred))])
    return (A - C) / (B - C)
#end cup

