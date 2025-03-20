import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from sklearn.cluster import DBSCAN
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm


def adjusted_r2(y_true, y_pred, num_predictors):
    n = len(y_true)  # Number of observations
    r2 = r2_score(y_true, y_pred)  # Calculate R-squared
    return 1 - ((1 - r2) * (n - 1) / (n - num_predictors - 1))


def theta_to_r(theta, distance):
    
    r=distance*np.tan(theta)
    return r

def homoscedasticity_test(X_test, residuals):
    # Add a constant to the independent variables (required for some statsmodels tests)
    X_test_with_const = sm.add_constant(X_test)

    print(len(X_test_with_const))
    print(len(residuals))

    # Perform Breusch-Pagan test
    bp_test_stat, bp_p_value, _, _ = het_breuschpagan(residuals, X_test_with_const)
    print("Breusch-Pagan Test Statistic:", bp_test_stat)
    print("Breusch-Pagan p-value:", bp_p_value)
    

    # Optional: White's Test (similar procedure, using statsmodels)
    white_test_stat, white_p_value, _, _ = het_white(residuals, X_test_with_const)
    print("White Test Statistic:", white_test_stat)
    print("White Test p-value:", white_p_value)
    return 0


def check_for_overlap(data1,data2):
    """
    Function to check if two pd data sets taken from on have the same indices
    i.e. check for no overlap in test and train data

    Inputs:
    data1: part of a pd dataframe
    data2: another part of the same pd dataframe

    Returns:
    True: If no overlap
    False: If overlap
    """
    data1_indices = data1.index.tolist()
    data2_indices = data2.index.tolist()
    # Check if the two lists have no numbers in common
    no_overlap = set(data1_indices).isdisjoint(set(data2_indices))
    return no_overlap



# Load your data
data_file_path = "Processed_Data/data_sets/output_test_600.csv"
df = pd.read_csv(data_file_path, comment='#')
df['R'] = df['Mean Theta'].apply(lambda theta: theta_to_r(theta, 11))

# Select features (X) and target (y)
X = df[["R",'X-ray Critical Energy', 'X-ray Percentage']] 

y = df['Emittance']  # target column name


pipeline = Pipeline([
    ('scaler', StandardScaler()),         # Step 1: Scale the features
    ('model', linear_model.LinearRegression())         # Step 2: Fit a linear regression model
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=45)

print("No overlap:",check_for_overlap(X_train,X_test))


# Fit the model to the training data
pipeline.fit(X_train, y_train)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
# Predict emittance values from test data
y_pred = pipeline.predict(X_test)
y_pred_train=pipeline.predict(X_train)


# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
mse_train=mean_squared_error(y_train,y_pred_train)
# Calculate R-squared
r2 = r2_score(y_test, y_pred)
r2_train=r2_score(y_train,y_pred_train)
# Calculate adjusted R-squared
num_predictors = X.shape[1]  # Number of predictors
adj_r2 = adjusted_r2(y_test, y_pred, num_predictors)
adj_r2_train=adjusted_r2(y_train,y_pred_train,num_predictors)

# Print the results
print("Mean Squared Error (test):", mse)
print("R-squared(test):", r2)
print("Adjusted R-squared values(test):",adj_r2)

print("Mean Squared Error (train):", mse_train)
print("R-squared(train):", r2_train)
print("Adjusted R-squared values(train):",adj_r2_train)

# Access the coefficients of the linear regression model
coefficients = pipeline.named_steps['model'].coef_
intercept = pipeline.named_steps['model'].intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Calculate upper and lower bounds
#mse=mse+(mse*0.1515)



x=np.linspace(min(min(y_pred), min(y_test)),max(max(y_pred), max(y_test)),100)
y_upper = x + np.sqrt(mse)
y_lower = x - np.sqrt(mse)

y_upper3 = x + np.sqrt(mse)*3
y_lower3 = x - np.sqrt(mse)*3

x_error=np.linspace(mse,mse,len(y_pred))

#------------------------------------------------------------------------

# Calculate residuals
residuals =y_pred-y_test

# Create figure and GridSpec
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)

# Main scatter plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_pred, label="Emittance data", color='tab:blue')
ax1.plot(x, x, color='k', label=r"$y=\^y$")
ax1.fill_between(x, y_lower, y_upper, color="red", alpha=0.2, label=(r"$\sqrt{\text{MSE}}$: "+str(np.round(np.sqrt(mse),3))))
ax1.set_ylabel(r"LR model prediction for emittance ($\mu m$)", fontsize=14)
ax1.set_xlabel(r"QV3D data values for emittance ($\mu m$)", fontsize=14)
ax1.set_title(r"Emittance predicted by model vs QV3D simulation", fontsize=16)
ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.legend(fontsize=12)
ax1.tick_params(axis='both', labelsize=12)

# Residuals plot (in units of sigma)
ax2 = fig.add_subplot(gs[1, 0])
ax2.errorbar(y_test, residuals/np.sqrt(mse), color='tab:blue', alpha=0.7, fmt='o',label="Residuals")

ax2.axhline(0, color='k', linestyle='--', linewidth=1)
ax2.axhline(-1,color='r',linestyle='--',linewidth=1)
ax2.axhline(1,color='r',linestyle='--',linewidth=1)
ax2.set_ylabel(r"Residuals ($\sigma$)", fontsize=14)
ax2.set_xlabel(r"QV3D data values for emittance ($\mu m$)", fontsize=14)
ax2.set_ylim(-np.max(np.abs(1.1*residuals/np.sqrt(mse))), np.max((np.abs(1.1*residuals/np.sqrt(mse)))))


plt.show()

#normality tests---------------------------------------------------------------------------------------------------
# Define the lower and upper bounds
lower_bound = -np.sqrt(mse)  # Residuals greater than this
upper_bound = np.sqrt(mse)  # Residuals less than this

# Count the number of residuals within the range
count_within_range = np.sum((residuals > lower_bound) & (residuals < upper_bound))

# Output the result
print(f"% of points within the range {lower_bound} < residuals < {upper_bound}: {(count_within_range/len(residuals))*100}%")


# Calculate excess kurtosis
data = residuals  # Replace with your data
excess_kurtosis = stats.kurtosis(data, fisher=True)  # Fisher's definition subtracts 3 from kurtosis

print(f"Excess Kurtosis: {excess_kurtosis}")
if excess_kurtosis > 0:
    print("The data has heavy tails (fat tails).")
elif excess_kurtosis < 0:
    print("The data has light tails.")
else:
    print("The data has normal tails.")

skewness=stats.skew(data)
print(f"Residual Skewness: {skewness}")

# Perform the Shapiro-Wilk Test
statistic, p_value = stats.shapiro(residuals)

# Output the results
print(f"Shapiro-Wilk Test Statistic: {statistic}")
print(f"P-value: {p_value}")

# Interpretation of the p-value
if p_value > 0.05:
    print("The data is likely normally distributed (fail to reject H0).")
else:
    print("The data is likely not normally distributed (reject H0).")


plt.hist(residuals, bins=20)
# plt.show()


# Assuming residuals is a NumPy array of residual values
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)

# Compute Z-scores
z_scores =np.array( (residuals - mean_residual) / std_residual)

# Identify outliers
y_test_array = y_test.values

outliers = np.where(np.abs(z_scores) > 3)

print(f"Outlier indices: {outliers[0]}")
print("Z-score(s) of outliers",z_scores[outliers[0]])
print("y test data",y_test_array[outliers[0]])
print("y prediction data",y_pred[outliers[0]])

# Compute Q1 and Q3
Q1 = np.percentile(residuals, 25)
Q3 = np.percentile(residuals, 75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))
print(f"Outlier indices: {outliers}")
print("y test data",y_test_array[outliers[0]])
print("y prediction data",y_pred[outliers[0]])


# Assuming residuals is a 1D array; reshape for DBSCAN
residuals_reshaped = residuals.to_numpy().reshape(-1, 1)

# Apply DBSCAN
dbscan = DBSCAN(eps=1, min_samples=5)  # Adjust eps and min_samples as needed
labels = dbscan.fit_predict(residuals_reshaped)

# Identify outliers (points with label -1)
outliers = np.where(labels == -1)
print(f"Outlier indices: {outliers}")

# Homoscasticity test:
homoscedasticity_test(X_test, residuals)

#VIF------------------------------------
# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns

# Calculate VIF
#vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF results
print(vif_data)

