import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

df=pd.read_csv("loan.csv")

# print(df.head())

# print(df.shape)

# print(df.isnull().sum())

# print(df.describe())

# print(df.info())

# print(df.duplicated().sum())

df=df.drop('Loan_ID',axis=1)
# convert the categorical data into numerica data
df['Gender']=df['Gender'].map({
    'Male':0,
    'Female':1
})


df['Married']=df['Married'].map({
    'No':0,
    'Yes':1
})

df['Dependents']=df['Dependents'].map({
    '0':0,
    '1':1,
    '2':2,
    '3+':3
})
df['Education']=df['Education'].map({
    'Not Graduate':0,
    'Graduate':1
})
df['Self_Employed']=df['Self_Employed'].map({
    'No':0,
    'Yes':1
})


df['Loan_Status']=df['Loan_Status'].map({
    'N':0,
    'Y':1
})

df['Property_Area']=df['Property_Area'].map({
    'Rural':0,
    'Semiurban':1,
    'Urban':2
})

df=df.drop_duplicates()

print("After preprocessing")
print(df.head(2))


X=df.drop('Loan_Status',axis=1)
y=df['Loan_Status']

x_test,x_train,y_test,y_train=train_test_split(
    X,y,test_size=0.2,random_state=24,
    stratify=y
)

model=DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("\nPerformance Matrix")

print("Accuracy Score: ",accuracy_score(y_test,y_pred))
print("Classification Report: ",classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
sns.heatmap(
    cm,
    cmap="Blues",
    fmt='d',
    annot=True,
    yticklabels=['Eligible','Not Eligible'],
    xticklabels=['Eligible','Not Eligible']
)

plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(12,8))
# Decision Tree
plot_tree(
    model,
    feature_names=X.columns,
    rounded=True,
    fontsize=9,
    filled=True
)

plt.title("Decision Tree")
plt.xlabel("Feature name")
plt.ylabel("height")
plt.show()