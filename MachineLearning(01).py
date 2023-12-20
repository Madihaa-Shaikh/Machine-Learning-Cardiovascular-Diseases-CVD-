#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("hello")


# In[68]:


import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Assuming cardio_train.csv is in the same directory as your Python script
df = pd.read_csv('cardio_train.csv',sep=';')
df


# In[3]:


print(df.shape)


# In[5]:


print(df.columns)


# In[44]:


# Assuming 'df' is your dataframe
age_values = df['age'].values

# Compute the year of the agevalues
year_of_birth = age_values/365


# In[45]:


print (year_of_birth)


# In[47]:


mean_age = year_of_birth.mean()


# In[48]:


print (mean_age)


# In[49]:


# Compute maximum and minimum of the age values
max_age =year_of_birth.max()
min_age =year_of_birth.min()
print(f"The maximum age is: {max_age} years")
print(f"The minimum age is: {min_age} years")


# In[57]:


# Assuming 'data' is your DataFrame with features and target

# Define features (X) and target (y)
X = df.drop(['id', 'cardio'], axis=1)  # Assuming 'id' is not a feature, and 'cardio' is the target
y = df['cardio']


# In[58]:


# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[59]:


# Scale data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[61]:


# Create a Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
logistic_regression_model.fit(X_train_scaled, y_train)


# In[62]:


# Predict the output for the test set
y_pred = logistic_regression_model.predict(X_test_scaled)

# Optionally, you can also calculate the predicted probabilities
y_pred_proba = logistic_regression_model.predict_proba(X_test_scaled)


# In[63]:


# Compare predicted output with real values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Print the comparison
print(comparison)


# In[66]:


# Check if the model is Logistic Regression
if hasattr(logistic_regression_model, 'coef_'):
    # Get feature names
    feature_names = X.columns

    # Get coefficients
    coefficients = logistic_regression_model.coef_[0]

    # Combine feature names and their coefficients
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})

    # Sort by importance
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    print("Feature Importance:")
    print(feature_importance)
else:
    print("This model does not have feature importances.")


# In[67]:


# Assuming 'X' is your feature DataFrame

# Select one example (proband)
proband_example = X.iloc[0]  # Assuming you want the first example

# Print features and their values for the proband
for feature, value in proband_example.items():
    print(f"{feature}: {value}")


# In[69]:


# Set up the figure and axis
plt.figure(figsize=(10, 6))

# Create a bar chart
plt.bar(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')

# Add labels and title
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:




