### Home Value Analysis
# This project delves into predicting home values using Multiple Linear Regression. It aims to understand the train-test process of machine learning for accurate predictions, 
# evaluate model fit through various methods including kernel density plots, and introduce variable selection techniques such as Recursive Feature Selection and LASSO regression 
# to identify influential predictors. Through data analysis, this project aims to uncover the key factors affecting home value predictions.

# Importing necessary libraries
import pandas as pd  # Pandas for data manipulation
import numpy as np  # Numpy for numerical computations

# File name containing housing data
filename = "HousingData.csv"

# Reading the data from the CSV file into a pandas DataFrame
Housingdf = pd.read_csv(filename)

# Displaying the first few rows of the DataFrame
print(Housingdf.head())

# Printing concise summary information about the DataFrame
print(Housingdf.info())

# Printing the list of column names in the DataFrame
print(Housingdf.columns.tolist())

# Setting the display format for floating-point numbers to two decimal places
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Selecting a subset of columns from the DataFrame
data = Housingdf[['value', 'msa360', 'msa520', 'msa875', 'msa1120', 'msa1600', 'msa1680', 'msa1920',
                  'msa2160', 'msa2800', 'msa3360', 'msa3760', 'msa4480', 'msa5000', 'msa5080', 'msa5120',
                  'msa5380', 'msa5600', 'msa5640', 'msa5720', 'msa5775', 'msa6160', 'msa6200', 'msa6280',
                  'msa6780', 'msa7040', 'msa7240', 'msa7320', 'msa7360', 'msa7400', 'msa7600', 'msa8280',
                  'msa8840', 'metro', 'hown', 'hhblack', 'hinc', 'per', 'bedrms', 'built', 'leak', 'houseser',
                  'hhmov', 'fplwk']]

# Displaying descriptive statistics of the selected data
print(data.describe())

# Importing necessary modules for machine learning, model evaluation, and data visualization
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib import pyplot as plt

# Name for the Linear Regression model
lm = LinearRegression()

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Fitting the Linear Regression model
lm.fit(X_train, y_train)

# Creating a DataFrame of coefficients
coefficients = pd.concat([pd.DataFrame(X.columns), pd.DataFrame(np.transpose(lm.coef_))], axis=1)
coefficients.columns = ["variables", "Model 1 Coeff."]

# Making predictions
predicted_price = lm.predict(X_test)

# Evaluating predictions
mean_absolute_error = mae(y_test, predicted_price)
mean_squared_error = mse(y_test, predicted_price)
root_mean_squared_error = mean_squared_error ** 0.5
r_squared = r2_score(y_test, predicted_price)
adj_r = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

# Plotting Actual vs Fitted Values for Home Price
plt.figure(figsize=(15, 10))
ax1 = sns.kdeplot(data=y_test.squeeze(), fill=True, bw_method=0.25, color="purple", label="Actual Value")
sns.kdeplot(data=predicted_price.squeeze(), fill=True, bw_method=0.25, color="g", label="Fitted Value", ax=ax1)
plt.title('Figure 1. Actual vs Fitted Values for Home Price - Model 1')
plt.xlabel('Home Value')
plt.ylabel('Density')
plt.legend()
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
plt.show()

#Conducting variable selection using RFE (Recursive feature elimination)
from sklearn.feature_selection import RFE #Import necessary module for RFE
Transforming data using RFE
rfe_selector = RFE(lm, n_features_to_select=7, step=1)  #Use the lm model, select 7 features, remove one variable each time.
#Perform the selector on the training data.
rfe_selector = rfe_selector.fit(X_train, y_train)
# Report your results
selector_report = pd.concat([pd.DataFrame(X_train.columns), pd.DataFrame(np.transpose(rfe_selector.support_)), pd.DataFrame(np.transpose(rfe_selector.ranking_))], axis=1)
selector_report.columns = ["variables", "valuable?", "ranking"]
selector_report

#Transforming data using RFE
rfe_selector = RFE(lm, n_features_to_select=7, step=1)  #LM = Linear Model, 7 variables
rfe_selector = rfe_selector.fit(X_train, y_train)
selector_report = pd.concat([pd.DataFrame(X_train.columns), pd.DataFrame(np.transpose(rfe_selector.support_)), pd.DataFrame(np.transpose(rfe_selector.ranking_))], axis=1)
selector_report.columns = ["variables", "valuable?", "ranking"]
selector_report

#Creating a data frame for coefficents
coefficients2 = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(rfe_selector.estimator_.coef_))], axis = 1)
#Creating Column Names
coefficients2.columns = ["variables", "RFE 1 Coeff."]
# create a constant term data frame
constant_term2 = {"variables": ["constant"], "RFE 1 Coeff.": [rfe_selector.estimator.intercept_[0]]}
constant_df2 = pd.DataFrame(constant_term2)
#Creating Table
table2 = pd.concat([coefficients2, constant_df2])
table2.set_index("variables", inplace=True)
table2

##Testing the 2nd New Model
#New features dataframe containing only selected features through RFE
X_RFE = X_test[X_test.columns[rfe_selector.support_]]
lm.fit(X_RFE, y_test)

#Making predictions
predicted_price2 = lm.predict(X_RFE)

#Evaluating predictions
mean_absolute_error2 = mae(y_test, predicted_price2)
mean_squared_error2 = mse(y_test, predicted_price2)
root_mean_squared_error2 = mean_squared_error2**0.5
r_squared2 = r2_score(y_test, predicted_price2)
adj_r2 = 1 - (1-r_squared2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("MAE: {:.3f}".format(mean_absolute_error2))
print("MSE: {:.3f}".format(mean_squared_error2))
print("RMSE: {:.3f}".format(root_mean_squared_error2))
print("R-squared: {:.3f}".format(r_squared2))
print("adjusted R-squared: {:.3f}".format(adj_r2))

#Establishing plot
plt.figure(figsize=(15, 10))
#set up the kernel density plot. Note the first parameter is the data, the second is one which shades the area under the curve,
#the bw_method is a smoothing parameter, 
ax2 = sns.kdeplot(data = y_test.squeeze(), fill = True, bw_method = 0.5, color = "b", label="Actual Value")
#We need this plot the predicted_hours
sns.kdeplot(data = predicted_price2.squeeze(),fill = True, bw_method = 0.5,  color = "r", label="Fitted Values" , ax=ax2)
#Adding the labels
plt.title('Figure 2. Actual vs Fitted Values for Home Value - Model 2 RFE fixed number selected')
plt.xlabel('Price')
plt.ylabel('Density')
#Adding the legend
plt.legend()
plt.ticklabel_format(style='plain', axis='y') # to prevent scientific notation.
plt.ticklabel_format(style='plain', axis='x') # to prevent scientific notation.
plt.show()


#Number of features
nof_list=np.arange(1,len(X_test.columns))            
high_score=0
#Variable to store the optimum features
nof=0
option_list = []
adjscore_list =[]
for n in range(len(nof_list)):
    lm = LinearRegression()
    rfe_selector2 = RFE(lm,n_features_to_select=nof_list[n])
    rfe_selector2 = rfe_selector2.fit(X_train,y_train)
    X_rfe_train = X_train[X_train.columns[rfe_selector2.support_]]
    lm.fit(X_rfe_train,y_train)
    score = lm.score(X_rfe_train,y_train)
    adjscore = 1 - (1-score)*(len(y_train)-1)/(len(y_train)-X_rfe_train.shape[1]-1)
    option_list.append(nof)
    adjscore_list.append(adjscore)
    if(adjscore>high_score):
        high_score = adjscore
        nof = nof_list[n]

plt.figure(figsize=(15, 10))
plt.plot(option_list, adjscore_list, color = "r")
plt.title('Figure 3. Determining Optimal Number of Variables (Features) to Include')
plt.xlabel('Feature Number')
plt.ylabel('Score')
plt.xticks(np.arange(0, 40, 1))
plt.show()

selector_report2 = pd.concat([pd.DataFrame(X_train.columns), pd.DataFrame(np.transpose(rfe_selector2.support_)), pd.DataFrame(np.transpose(rfe_selector2.ranking_))], axis=1)
selector_report2.columns = ["variables", "valuable?", "ranking"]
selector_report2


#Coefficients for third model approach
coefficients3 = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(rfe_selector2.estimator_.coef_))], axis = 1)
coefficients3.columns = ["variables", "RFE 2 Coeff."]
#Creating constant term data frame
constant_term3 = {"variables": ["constant"], "RFE 2 Coeff.": [rfe_selector2.estimator.intercept_[0]]}
constant_df3 = pd.DataFrame(constant_term3)
#Creating table
table3 = pd.concat([coefficients3, constant_df3])
table3.set_index("variables", inplace=True)
table3

##Testing the 3rd New Model
#New features dataframe containing only selected features through RFE
X_rfe_test = X_test[X_test.columns[rfe_selector2.support_]]
lm.fit(X_rfe_test, y_test)

#Making predictions
predicted_price3 = lm.predict(X_rfe_test)

#Evaluating Predictions
mean_absolute_error3 = mae(y_test, predicted_price3)
mean_squared_error3 = mse(y_test, predicted_price3)
root_mean_squared_error3 = mean_squared_error3**0.5
r_squared3 = r2_score(y_test, predicted_price3)
adj_r3 = 1 - (1-r_squared3)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("MAE: {:.3f}".format(mean_absolute_error3))
print("MSE: {:.3f}".format(mean_squared_error3))
print("RMSE: {:.3f}".format(root_mean_squared_error3))
print("R-squared: {:.3f}".format(r_squared3))
print("adjusted R-squared: {:.3f}".format(adj_r3))


#Plotting Actual vs Fitted Values for Home Price
plt.figure(figsize=(15, 10))
#set up the kernel density plot. Note the first parameter is the data, the second is one which shades the area under the curve,
#the bw_method is a smoothing parameter, 
ax2 = sns.kdeplot(data = y_test.squeeze(), fill = True, bw_method = 0.5, color = "green", label="Actual Value")
#We need this plot the predicted_hours
sns.kdeplot(data = predicted_price3.squeeze(), fill = True, bw_method = 0.5, color = "red", label="Fitted Values" , ax=ax2)
#Adding the labels
plt.title('Figure 4. Actual vs Fitted Values for Home Values - Determining Optimal Features (Variables)')
plt.xlabel('Price')
plt.ylabel('Density')
#Adding the legend
plt.legend()
plt.ticklabel_format(style='plain', axis='y') # to prevent scientific notation.
plt.ticklabel_format(style='plain', axis='x') # to prevent scientific notation.
plt.show()


# #### Using LASSO Regression
from sklearn.linear_model import Lasso #Importing Lasso model data
lasso = Lasso(alpha=100)
# Fitting the Lasso model.
lasso.fit(X_train, y_train)
# Create the model score
lasso.score(X_test, y_test), lasso.score(X_train, y_train)
# Below we get the basic coefficients
lasso.coef_
#Coefficients for LASSO approach
coefficients4 = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(lasso.coef_))], axis = 1)
coefficients4.columns = ["variables", "Lasso Coeff."]
#Creating a constant term data frame
constant_term4 = {"variables": ["constant"], "Lasso Coeff.": [lasso.intercept_[0]]}
constant_df4 = pd.DataFrame(constant_term4)
#Creating table
table4 = pd.concat([coefficients4, constant_df4])
table4.set_index("variables", inplace=True)
table4
#Making predictions
predicted_price4 = lasso.predict(X_test)
#Evaluating predictions
mean_absolute_error4 = mae(y_test, predicted_price4)
mean_squared_error4 = mse(y_test, predicted_price4)
root_mean_squared_error4 = mean_squared_error4**0.5
r_squared4 = r2_score(y_test, predicted_price4)
adj_r4 = 1 - (1-r_squared4)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("MAE: {:.3f}".format(mean_absolute_error4))
print("MSE: {:.3f}".format(mean_squared_error4))
print("RMSE: {:.3f}".format(root_mean_squared_error4))
print("R-squared: {:.3f}".format(r_squared4))
print("adjusted R-squared: {:.3f}".format(adj_r4))

#Plotting Actual vs Fitted Values for Home Price for LASSO
plt.figure(figsize=(15, 10))
#set up the kernel density plot. Note the first parameter is the data, the second is one which shades the area under the curve,
#the bw_method is a smoothing parameter, 
ax2 = sns.kdeplot(data = y_test.squeeze(), fill = True, bw_method = 0.5, color = "green", label="Actual Value")
#We need this plot the predicted_hours
sns.kdeplot(data = predicted_price4.squeeze(), fill = True, bw_method = 0.5, color = "red", label="Fitted Values" , ax=ax2)
#Adding the labels
plt.title('Figure 5. Actual vs Fitted Values for Home Value - Lasso Results')
plt.xlabel('Price')
plt.ylabel('Density')
#Adding the legend
plt.legend()
plt.ticklabel_format(style='plain', axis='y') # to prevent scientific notation.
plt.ticklabel_format(style='plain', axis='x') # to prevent scientific notation.
plt.show()


# #### Collecting Results
# Following is a collection of all results for the purposes of comparison.

# Creating a Large Data Frame for All Models
Coeff_table = pd.concat([table, table2, table3, table4], axis=1) #This concatentates all previous results
Coeff_table.reset_index(inplace=True) #This resets the index for subsequent efforts at merging data

# Creating a dataframe with the performance metrics.
## Creating lists with the results of each model.
L1 = [mean_absolute_error, mean_squared_error, root_mean_squared_error, r_squared, adj_r]
L2 = [mean_absolute_error2, mean_squared_error2, root_mean_squared_error2, r_squared3, adj_r2]
L3 = [mean_absolute_error3, mean_squared_error3, root_mean_squared_error3, r_squared3, adj_r3]
L4 = [mean_absolute_error4, mean_squared_error4, root_mean_squared_error4, r_squared3, adj_r4]
## Creating column labels and make a list of lists of the four results.
L5 = ["MAE", "MSE", "RMSE", "R2", "ADJ_R"]
L6 = [L1, L2, L3, L4]
# Convert L6 into a dataframe.
Perf_df = pd.DataFrame(L6, columns = L5 )
# Transposing so this dataframe is lined up with the coefficients data frame.
Perf_dfT = Perf_df.transpose()
Perf_dfT.reset_index(inplace=True)
# Making sure that the column labels are the same between Perf_dfT and that of Coeff_table
Perf_dfT.rename(columns={"index":"variables", 0:"Model 1 Coeff.", 1:"RFE 1 Coeff.", 2:"RFE 2 Coeff.", 3:"Lasso Coeff."}, inplace=True)

# Appending the Performance Metrics to a Table
Coef_fin = pd.concat([Coeff_table, Perf_dfT])
#Coef_fin = Coeff_table.append(Perf_dfT, ignore_index=True)
Coef_fin.set_index("variables", inplace=True)
Coef_fin

#Exporting results to a csv to format for a document.
Coef_fin.to_csv('Model Results.csv')


