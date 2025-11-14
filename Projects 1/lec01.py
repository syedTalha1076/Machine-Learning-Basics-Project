# The below program check the versions of some common pythod libraries 

# import sklearn
# import matplotlib
# import pandas as pd
# import sys
# import scipy
# import numpy as np

# print("scikit-learn version:", sklearn.__version__)
# print("matplotlib version:", matplotlib.__version__)
# print("pandas version:", pd.__version__)
# print("sys version:", sys.version)
# print("scipy version:", scipy.__version__)
# print("numpy version:", np.__version__)

# ========================================================
#Program for iris dataset

# Load libraries
from tabnanny import check
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url,names=names) # panda use to load the data from url 
# columns and rows in dataset
print(dataset.shape)
# i want to print first 30 instances of my dataset
print(dataset.head(30))
# i want to get different attributes of my dataset
print(dataset.describe()) # all the values lies between 0 and 8 cm
# to get the size of each class
print("To get the size of each class\n")
print(dataset.groupby('class').size())

#? for visualizing the dataset we draw univariate plots and multivariate plots
#? "Uni" = One variable
#? These plots show the distribution or pattern of a single variable.
#? Useful for understanding:
#? Central tendency (mean, median),Spread (variance, range),Shape (normal, skewed),Outliers
print("box")
dataset.plot(kind = 'box',subplots = True, layout = (2,2), sharex = False, sharey = False) # sharex and sharey are used to share the x and y axis also check for True
plt.show()

# Histograms are used to visualize the distribution of a single variable
# hist is EDA (exploratory data analysis) tool and EDA is used to analyze the data
# and EDA is important for cleaning the data
print("Hist")
dataset.hist()
plt.show()

#? multvariate plots
#? These plots show the relationship between two or more variables
print("Scatter Matrix")
scatter_matrix(dataset)
plt.show()

#! Now its time to make some algorithms 
# we predict the accuracy based on the some unseen data
# Validation dataset is dataset which help to train the model so we use validation dataset
#*To use a validation dataset, you need to split your original dataset into three parts:
#Training set – to train the model
#Validation set – to tune and evaluate during training
#Test set – to evaluate the final model
#! Here we divide the dataset into 2 parts 80% for training and 20% for testing

array = dataset.values
x = array[:,0:4] # all the rows and first 4 columns
y = array[:,4] # all the rows and last column

validation_size = 0.20 # 20% of the dataset
seed = 6 # seed is used to get the same results every time we run the code
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=validation_size, random_state= seed)

#x_train, y_train	To train the model (learn the patterns)
#x_test, y_test	To test/validate the model (check how good it is)
# If you have 100 flowers:
# 80 go to training (x_train, y_train)
# 20 go to testing (x_test, y_test)

# To teach the model with some data (training) and then test it with new data (validation) to see if it learned correctly.

seed = 6
scoring = 'accuracy' # ratio of number of correct predictions to total predictions divided by total number of instances multiply by 100 to get percentage
# we use differnt algorithms here 
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []

for name, model in models:
   kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)

   cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
   results.append(cv_results)
   names.append(name)
   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   print(msg)

# The above code will print the accuracy of each model in which LDA is more accurate

