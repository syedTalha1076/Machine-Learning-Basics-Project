# Lecture 04 ==> About Classification
#? Decision Tree


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder # To convert String into numerical values(catagorical)
# from sklearn.model_selection import train_test_split # for test and train
# from sklearn.tree import DecisionTreeClassifier # for checking testing results
# from sklearn.metrics import classification_report, confusion_matrix # for visualizing tree and classification report use for prediction
# from sklearn.tree import plot_tree 

# irisV = sns.load_dataset('iris')
# print(irisV.head())
# print(irisV.info())
# print(irisV.shape)
# print(irisV.isnull().any())
# # print(irisV.isnull().sum())
# # Below graphe we understanding the reltionship or pattren between variables in dataset
# sns.pairplot(data=irisV,hue='species')
# plt.show()
# # Checking the corelation.
# sns.heatmap(irisV.select_dtypes(include='number').corr()) # higher shade it means thier will stroing corelatioon
# plt.show()

# #? Now we will separate the target variable(y) and features(X) as follows
# target = irisV['species']
# df1 = irisV.copy()
# df2 = df1.drop('species',axis=1)
# # defining  attributes
# x = df1.drop('species', axis=1)  # Only numeric features  # independent variable
# print(x)

# # Label encoding as target variable is cataogorical 
# # means convert setosa,versicolor and virginca into numeric values
# le = LabelEncoder()
# target = le.fit_transform(target)
# print(target) # 0 represent setosa , 1 ==> versicolor and 2==>virginica

# y = target

# # splitting the data
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # split the data into training and testing sets

# #defining the decision tree algorithm
# dtree = DecisionTreeClassifier()
# dtree.fit(X_train,y_train)
# print("decision Tree classifier created")

# #Predicting the value of test data
# y_prediction = dtree.predict(X_test)
# print("Classification report: \n",classification_report(y_test,y_prediction))

# from sklearn import tree
# fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
# cn=['setosa', 'versicolor', 'virginica']
# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
# tree.plot_tree(dtree,
#             feature_names = fn,
#             class_names= cn,
#             filled= True
# )

# fig.savefig("imagename.png")
# plt.show()


# # 👤 User Input for Prediction
# print("\n🌸 Let's predict the species for your flower!")

# # Ask user to enter values
# sepal_length = float(input("Enter sepal length (cm): "))
# sepal_width = float(input("Enter sepal width (cm): "))
# petal_length = float(input("Enter petal length (cm): "))
# petal_width = float(input("Enter petal width (cm): "))

# # Store values in correct shape for prediction
# new_sample = [[sepal_length, sepal_width, petal_length, petal_width]]

# # Predict using trained model
# predicted_class = dtree.predict(new_sample)[0]

# # Convert number back to label
# species_label = le.inverse_transform([predicted_class])[0]

# # Show the result
# print(f"\n✅ The predicted species is: **{species_label.upper()}** 🌺")
# print("Classification report: \n",classification_report(y_test,y_prediction))

# ===========================================================================
# ===========================================================================

#!   Random Forest

# import pandas as pd
# import numpy as np
# import seaborn as sns

# # Load the data
# df = sns.load_dataset('penguins')
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.isnull().sum())

# #drop the null values 
# print("Droping null values")
# print(df.dropna(inplace=True))
# print(df.isnull().sum())

# #? Transforming the data into numeric values==> This is called one hot encoding
# print(df.sex.unique())
# print(pd.get_dummies(df['sex']).head())

# # dropping first column(female column) because we already knows if 1 it means punguin is male if 0 means female so we 
# # we can identify from one column no need of another column
# print("Female Column drops")
# sex = pd.get_dummies(df['sex'],drop_first=True)
# print(sex.head())

# #? Transforming the data into numeric values==> This is called one hot encoding
# # for island
# print(df.island.unique())
# print(pd.get_dummies(['island']).head())

# island = pd.get_dummies(df['island'],drop_first=True)
# print(island.head())

# #* Concantenate the above two data fram into original df
# print("After Cancantenation")
# new_data = pd.concat([df,island,sex],axis=1)
# print(new_data.head())

# # drop the repeated column
# new_data.drop(['sex','island'],axis=1, inplace=True)
# print(new_data.head())

# #? Create a separate target variable
# print("Create a separate target variable")
# y = new_data.species
# print(y.head())
# print(y.unique())
# y = y.map({'Adelie':0,'Chinstrap':1,'Gentoo':2}) # map function is used to convert into numeric value
# print(y.head()) 

# #? Dropping the target variale: Species
# print(" Dropping the target variale: Species")
# new_data.drop('species',inplace=True,axis=1)
# print(new_data)

# x = new_data

# #? Performing spltting data
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  # split the data into training and testing sets
# print("X_train: ",X_train.shape)
# print("X_test: ",X_test.shape)
# print("Y_train: ",y_train.shape)
# print("Y_test: ",y_test.shape)

# #* Training Random forest classification on training set
# #* Training Random Forest Classifier on training set
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Initialize the model
# model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)

# # Train the model
# model.fit(X_train, y_train)

# # Predict on test set
# y_pred = model.predict(X_test)

# # Evaluate performance
# print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
# print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
# # ==========================================
# # 🎯 USER-DEPENDENT PREDICTION (NO WARNING)
# # ==========================================

# print("\n🔍 Now enter penguin features to predict its species:")

# bill_length = float(input("Enter bill_length_mm: "))
# bill_depth = float(input("Enter bill_depth_mm: "))
# flipper_length = float(input("Enter flipper_length_mm: "))
# body_mass = float(input("Enter body_mass_g: "))

# # Island input (one-hot encoded: only Dream and Torgersen; Biscoe is base)
# island_input = input("Enter island (Biscoe, Dream, Torgersen): ").strip().capitalize()
# island_dream = 1 if island_input == "Dream" else 0
# island_torgersen = 1 if island_input == "Torgersen" else 0

# # Sex input (Male or Female)
# sex_input = input("Enter sex (Male/Female): ").strip().capitalize()
# male = 1 if sex_input == "Male" else 0

# # Create DataFrame with correct feature names
# user_input = pd.DataFrame([{
#     'bill_length_mm': bill_length,
#     'bill_depth_mm': bill_depth,
#     'flipper_length_mm': flipper_length,
#     'body_mass_g': body_mass,
#     'Dream': island_dream,
#     'Torgersen': island_torgersen,
#     'Male': male
# }])

# # Predict species
# prediction = model.predict(user_input)
# species_map = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
# print(f"🧠 Predicted Species: {species_map[prediction[0]]}")

# ================================================================
# ================================================================

# ! KNN Algorithm

# import pandas as pd
# import numpy as np
# import matplotlib.pylab as plt
# import seaborn as sns
# from sklearn.datasets import load_breast_cancer

# cancer = load_breast_cancer()
# print(cancer.keys())
# # print(cancer['DESCR'])
# # print(cancer['feature_names'])

# # set up data frame
# df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
# print(df_feat.info())

# # Check the traget value which tells us about milagnant(cancer) or beining(not cancer)
# # 0-> cancer and 1 -> not cancer
# # print(cancer['target'])
# #?Convert target into dataframe
# df_target = pd.DataFrame(cancer['target'],columns=['cancer'])
# print(df_feat.head())

# # Standarizing variable
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# scaler.fit(df_feat)
# scaled_feature = scaler.fit_transform(df_feat)
# # It converts your scaled data (NumPy array) back into a Pandas DataFrame, 
# # and gives it the original column names
# df_feat_scaled = pd.DataFrame(scaled_feature,columns=df_feat.columns)
# print(df_feat_scaled.head())

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(scaled_feature, np.ravel(df_target), test_size=0.3, random_state=105)  # split the data into training and testing sets
# # test_size = 0.30 it means 70% for training and 30% for testing data

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=1)
# '''You're creating a K-Nearest Neighbors classifier where K = 1 — so 
#  for any new data point, the model will look at the 1 closest 
#  neighbor in the training set to decide its class.'''

# knn.fit(X_train,y_train)
# pred = knn.predict(X_test)

# from sklearn.metrics import classification_report,confusion_matrix
# print(confusion_matrix(y_test,pred))
# print(classification_report(y_test,pred))

# #* Choose K values
# error_rate = []
# for i in range(1,40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train,y_train)
#     pred_i = knn.predict(X_test)
#     error_rate.append(np.mean(pred_i != y_test))

# plt.figure(figsize=(10,6))
# plt.plot(
#     range(1,40),
#     error_rate,
#     color = 'blue',
#     linestyle = 'dashed',
#     marker = 'o',
#     markerfacecolor = 'red',
#     markersize = 10
#     ) 
# plt.title('Error rate vs K value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()

# # the best value for k is 21 because see from graph at 21 the error values is small
# # In the above change n_neigbour=1 from 1 to 21 see accuracy imporove to 0.99 check below
# # Also change confusion matrix

# knn1 = KNeighborsClassifier(n_neighbors=21)
# knn1.fit(X_train,y_train)
# pred = knn1.predict(X_test)
# print(confusion_matrix(y_test,pred))
# print(classification_report(y_test,pred))

# # user dependent
# print("\n🔍 Let's Predict Cancer Based on User Input")

# # 🚨 Replace these values with user-provided values
# r = 17.0    # mean radius
# t = 10.5    # mean texture
# p = 115.0   # mean perimeter
# a = 980.0   # mean area
# s = 0.1     # mean smoothness

# # Create dataframe from simulated user input
# user_input = pd.DataFrame([[r, t, p, a, s]], columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])

# # Scale user input using the same scaler
# all_columns = df_feat.columns
# user_data = dict.fromkeys(all_columns, 0)  # fill with 0 by default

# # Set the values you have
# user_data['mean radius'] = r
# user_data['mean texture'] = t
# user_data['mean perimeter'] = p
# user_data['mean area'] = a
# user_data['mean smoothness'] = s

# # Create DataFrame with all columns in correct order
# user_input = pd.DataFrame([user_data])

# # Now transform
# user_scaled = scaler.transform(user_input)
# # Predict
# user_pred = knn1.predict(user_scaled)

# # Output
# if user_pred[0] == 0:
#     print("⚠️  Prediction: Cancer Detected (Malignant)")
# else:
#     print("✅ Prediction: No Cancer (Benign)")


# ==================================================================
# ===================================================================

#? Another program on KNN for iris dataset

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, classification_report

# # 🌐 Load the Iris dataset from UCI
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
# df = pd.read_csv(url, names=columns)

# print(df.head())
# print(df['species'].value_counts())

# # 🎯 Encode the target variable
# df['species'] = df['species'].map({
#     'Iris-setosa': 0,
#     'Iris-versicolor': 1,
#     'Iris-virginica': 2
# })

# # ✅ Features and target
# X = df.drop('species', axis=1)
# y = df['species']

# # 🔄 Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 🔀 Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.3, random_state=42
# )

# # 🧠 Initial KNN model (K=1)
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, y_train)
# pred = knn.predict(X_test)

# # 📊 Evaluation
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
# print("\nClassification Report:\n", classification_report(y_test, pred))

# # 🔍 Find best K value
# error_rate = []
# for i in range(1, 40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     error_rate.append(np.mean(pred_i != y_test))

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed',
#          marker='o', markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()

# # 👑 Best K found manually (e.g. K=11)
# knn_best = KNeighborsClassifier(n_neighbors=11)
# knn_best.fit(X_train, y_train)
# final_pred = knn_best.predict(X_test)
# print("\n📌 Final Model Accuracy Report (K=11)")
# print(classification_report(y_test, final_pred))

# # 👤 User-Dependent Input
# print("\n🔍 Predict Iris Species Based on Your Input")
# sl = float(input("Enter sepal length (cm): "))
# sw = float(input("Enter sepal width (cm): "))
# pl = float(input("Enter petal length (cm): "))
# pw = float(input("Enter petal width (cm): "))

# # Make user input into DataFrame
# user_df = pd.DataFrame([[sl, sw, pl, pw]], columns=X.columns)
# user_scaled = scaler.transform(user_df)

# # Predict the class
# user_pred = knn_best.predict(user_scaled)

# # Map back to species name
# species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
# print("🌼 Predicted Species:", species_map[user_pred[0]])
# ====================================================
# ===================================================

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import requests
# from PIL import Image
# from io import BytesIO

# #? Iris setosa
# # url1 = "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg"  # Example direct image URL
# # response = requests.get(url1)
# # img1 = Image.open(BytesIO(response.content))
# # plt.imshow(img1)
# # plt.axis('off')
# # plt.show()

# #? iris versicolor
# # url2 = "https://upload.wikimedia.org/wikipedia/commons/7/7a/Iris_versicolor.jpg"
# # response = requests.get(url2)
# # img2 = Image.open(BytesIO(response.content))
# # plt.imshow(img2)
# # plt.axis('off')
# # plt.show()

# #? iris verginica
# # url3 ="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/500px-Iris_virginica_2.jpg"
# # response = requests.get(url3)
# # img3 = Image.open(BytesIO(response.content))
# # plt.imshow(img3)
# # plt.axis('off')
# # plt.show()

# iris = sns.load_dataset('iris')
# print(iris.head())

# #Exploratory data anaylsis
# sns.pairplot(iris,hue='species',palette='Dark2')
# plt.show()

# # create a plot for sepalLength and sepalWidth for setosa
# setosa = iris[iris['species'] == 'setosa']

# sns.kdeplot(
#     x=setosa['sepal_length'],
#     y=setosa['sepal_width'],
#     cmap='plasma',
#     fill=True  
# )

# plt.title('2D KDE Plot for Setosa')
# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.show()

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# # Step 1: Preprocessing (Standardize the features)
# x = iris.drop('species', axis=1)
# y = iris['species']

# scaler = StandardScaler()
# scaler.fit(x)
# x_scaled = scaler.transform(x)

# # Step 2: Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

# # Step 3: Model Training (Initial Model)
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, y_train)
# pred = knn.predict(X_test)

# # Step 4: Evaluation
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, pred))
# print("\nClassification Report:")
# print(classification_report(y_test, pred))
# print("Accuracy:", accuracy_score(y_test, pred))

# # Step 5: Error Rate for Various K values
# error_rate = []

# for i in range(1, 40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     error = np.mean(pred_i != y_test)
#     error_rate.append(error)

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red')
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()

# # Step 6: Best K Value (e.g. K=9 if error rate is lowest around 9)
# best_k = error_rate.index(min(error_rate)) + 1
# knn_best = KNeighborsClassifier(n_neighbors=best_k)
# knn_best.fit(X_train, y_train)
# pred_best = knn_best.predict(X_test)

# print(f"\nUsing K = {best_k} gives the best performance")
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, pred_best))
# print("Classification Report:")
# print(classification_report(y_test, pred_best))
# print("Accuracy:", accuracy_score(y_test, pred_best))

# # Step 7: User Input Prediction (Example)
# print("\n\U0001F50D Let's Predict Species Based on User Input")
# s1 = float(input("Enter Sepal Length (cm): "))
# s2 = float(input("Enter Sepal Width (cm): "))
# p1 = float(input("Enter Petal Length (cm): "))
# p2 = float(input("Enter Petal Width (cm): "))

# user_input = pd.DataFrame([[s1, s2, p1, p2]], columns=iris.columns[:-1])
# user_scaled = scaler.transform(user_input)
# user_pred = knn_best.predict(user_scaled)

# print("\n\U0001F4A1 Predicted Species:", user_pred[0])

# ===================================================================
# ===================================================================

#! Naives Bayes Theorem

# import pandas as pd
# from sklearn.naive_bayes import MultinomialNB

# df = pd.read_csv('playing.csv')
# # print(df.head())
# #check which values are in cataogrical form
# print(df.info())
# # Convert into catogrical form
# df = df.apply(lambda x : x.astype('category'))
# # print(df.head())
# # convert into 0s and 1s
# df1 = df.apply(lambda x : x.cat.codes)
# # print(df1)

# #? Dividing dataframe into training and testing

# train = df1[:10]
# test = df1[-4:] # selecting last 4 rows(containing outook,temp,humidity and windy)

# y_train = train.pop('Play')
# x_train = train

# y_test = test.pop('Play')
# x_test = test

# #? X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #? we can also use the above the line instead of train and testing manaully


# model = MultinomialNB()
# model_obj = model.fit(x_train,y_train)
# prediction = model.predict(x_test)
# print("Prediction: ",prediction)

# from sklearn.metrics import accuracy_score,classification_report

# print("Accuracy: ",accuracy_score(y_test,prediction))
# print("Classifiction report:\n ",classification_report(y_test,prediction))

# print("\n📥 Enter today's weather to check if you can play or not.")

# # Mapping inputs to numbers manually (must match the training encoding)
# # You can see mappings by printing: df.apply(lambda x: x.cat.categories)

# # Ask user for input (Keep values exactly same as in dataset)
# outlook = input("Outlook (Sunny / Overcast / Rain): ").strip().title()
# temperature = input("Temperature (Hot / Mild / Cool): ").strip().title()
# humidity = input("Humidity (High / Normal): ").strip().title()
# windy = input("Windy (True / False): ").strip().title()

# # Manual mappings (based on the order in dataset)
# outlook_map = {'Overcast': 0, 'Rain': 1, 'Sunny': 2}
# temp_map = {'Cool': 0, 'Hot': 1, 'Mild': 2}
# humidity_map = {'High': 0, 'Normal': 1}
# windy_map = {'False': 0, 'True': 1}

# # Convert user input into DataFrame
# user_data = pd.DataFrame([{
#     'Outlook': outlook_map.get(outlook, -1),
#     'Temperature': temp_map.get(temperature, -1),
#     'Humidity': humidity_map.get(humidity, -1),
#     'Windy': windy_map.get(windy, -1)
# }])

# # Check for any invalid values
# if -1 in user_data.values:
#     print("❌ Invalid input. Please enter values exactly as shown.")
# else:
#     user_pred = model.predict(user_data)
#     if user_pred[0] == 1:
#         print("✅ You can play!")
#     else:
#         print("❌ Sorry, you should not play today.")

# ================================================