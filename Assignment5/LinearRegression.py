import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

data=pd.read_csv("temperatures.csv")

months=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

X=data["YEAR"]

x_train,x_test=train_test_split(X,test_size=0.25,random_state=42)
models = {}

for month in months:
    y=data[month]
    
    y_train=y.loc[x_train.index]
    y_test=y.loc[x_test.index]
    
    model=LinearRegression()
    model.fit(x_train,y_train)
    models[month] = model
    
    y_pred=model.predict(x_test)
    
    print("Month: ",month)
    print("Mean Squared error : ",mean_squared_error(y_test,y_pred))
    print("Mean Absolute error : ",mean_absolute_error(y_test,y_pred))
    print("R-squared : ",r2_score(y_test,y_pred))
    
#predict the temperature for user input year
year=int(input("Enter the year for prediction: "))
month=input("Enter the month : ").upper()

if month in months:
    
    y_pred=models[month].predict(pd.DataFrame({'YEAR':[year]}))[0]
    print("Predicted temperature ",year,"for month ",month,"is : ",y_pred)
else:
    print("Invalid month")
    
# Visualize January temperature
model = LinearRegression()
model.fit(X, data['JAN'])

plt.scatter(data['YEAR'], data['JAN'], label="Actual Temperature")
plt.plot(data['YEAR'], model.predict(X), color='red', label="Regression Line")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.title("January Temperature Prediction")
plt.legend()
plt.show()



