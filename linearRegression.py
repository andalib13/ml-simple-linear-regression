import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#%matplotlib inline

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)
print(df.sample(5))

print(df.info())
print(df.describe())
print(df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']].head(9))

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head(9))
cdf.hist()
#plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FLUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
#plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("ENGINE SIZE")
plt.ylabel("CO2 EMISSIONS")
#plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("CYLINDERS")         
plt.ylabel("CO2 EMISSIONS")
plt.show()

x= cdf.ENGINESIZE.to_numpy()
y=cdf.CO2EMISSIONS.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# print(x_train)
# print(y_train)

type(x_train), np.shape(x_train), type(y_train), np.shape(y_train)

regressor = linear_model.LinearRegression()
regressor.fit(x_train.reshape(-1,1), y_train)
print('Coefficients: ', regressor.coef_)
print('Intercept: ', regressor.intercept_)  

plt.scatter(x_train, y_train,  color='blue')
plt.plot(x_train, regressor.coef_ * x_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

y_test_ = regressor.predict(x_test.reshape(-1,1))
                            
                            # Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))

#regression with test data
plt.scatter(x_test, y_test,  color='green')
plt.plot(x_test, regressor.coef_* x_test + regressor.intercept_, '-b')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# repeat with fule consumption instead of engine size
x= cdf.FUELCONSUMPTION_COMB.to_numpy()      
y=cdf.CO2EMISSIONS.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
regressor = linear_model.LinearRegression()
regressor.fit(x_train.reshape(-1,1), y_train)   
print('Coefficients: ', regressor.coef_)
print('Intercept: ', regressor.intercept_)
plt.scatter(x_train, y_train,  color='blue')
plt.plot(x_train, regressor.coef_ * x_train + regressor.intercept_, '-r')
plt.xlabel("FUEL CONSUMPTION")  
plt.ylabel("EMISSION")
plt.show()
