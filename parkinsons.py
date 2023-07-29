import numpy as np 
import pandas as pd 
from IPython.display import Image 
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, auc
import os




for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
Image('/kaggle/input/parkinson/Header.jpg', width=1200)
print(Image)
data=pd.read_csv("parkinsons.data")
print(data)
print(data["status"].value_counts())
data.hist(bins=50, figsize =(20,15), color = 'darkslategrey')

minmax = MinMaxScaler()
data_boxplot = data.drop(['name'],axis=1)
data_boxplot = minmax.fit_transform(data_boxplot)
boxplot = pd.DataFrame(data_boxplot, columns = data.drop(['name'],axis=1).columns)
boxplot.boxplot(figsize=(30,14))

plt.figure(figsize=(30,16))
sns.heatmap(data.corr(), annot = True, cmap = 'Greens',annot_kws={'size':19})
plt.show()
