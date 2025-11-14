
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('SUV.csv')
print(dataset.head(10))
# Iloc function n pandas is used for integer-location based indexing to select rows and columns by their integer positions (not by labels).
x = dataset.iloc[:,2:4].values # : means all rows and 2:3 means columns means i want all rows and 2 aand 3 column
y = dataset.iloc[:,4].values

print("Independent Variable: ",x)
print("Dependent Variable: ",y)

# Training and Testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  # split the data into training and testing sets

from sklearn.preprocessing import StandardScaler
# StandardScaler make data Standard which has mean = 0 and standard deviation = 1
# it helps model converge faster and prevents features with larger values form dominatng

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

prediction = classifier.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,prediction)*100,"%") # want accuracy in percentage thats why multiply by 100
# now predict for new people
age = int(input("Enter age: "))
salary = int(input("Enter estimated salary: "))

person = [[age, salary]]
person_scaled = sc.transform(person)
result = classifier.predict(person_scaled)

if result[0] == 1:
    print("This person will likely BUY an SUV.")
else:
    print("This person will NOT buy an SUV.")