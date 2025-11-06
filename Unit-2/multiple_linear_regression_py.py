import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. Load Data
df = pd.read_csv('../DataSets/50_Startups.csv')

# 2. Define Features (X) and Target (y)
# We do this FIRST, on the raw, un-scaled data.
X = df.drop('Profit', axis=1) # X is everything except the target
y = df['Profit']             # y is just the target

# 3. Split the data
# This is the MOST IMPORTANT first step after defining X and y.
# We split the raw data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define Preprocessing
# Identify which columns are which type
numeric_features = ['R&D Spend', 'Administration', 'Marketing Spend']
categorical_features = ['State']

# Create a "transformer" for numeric columns
# It will apply StandardScaler
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create a "transformer" for categorical columns
# It will apply OneHotEncoder and drop one category

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Use ColumnTransformer to combine these two transformers
# It will apply the right transformer to the right columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Create the Full Model Pipeline
# This pipeline first runs the preprocessor, then runs the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 6. Train the Model
# We fit the *entire* pipeline on the raw X_train and y_train.
# Scikit-learn automatically handles all the fitting and transforming
# *only* on the training data. No leakage!
model.fit(X_train, y_train)

# 7. Make Predictions
# We predict on the raw X_test. The pipeline automatically
# applies the *already-fitted* transformations.
y_pred = model.predict(X_test)

# 8. Evaluate Metrics (on the *original* scale)
# Since we never scaled y, our y_test and y_pred are already
# in the correct, interpretable units (e.g., Dollars).
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"R-squared (R2): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# 9. Correct Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs. Predicted Profit (Test Set)")
plt.grid(True)
plt.show()

# 10. Inspect Coefficients (Optional)
# This is how you see the coefficients from the pipeline
regressor = model.named_steps['regressor']
preprocessor = model.named_steps['preprocessor']

# Get feature names from the preprocessor
# It's a bit complex, but good to know
cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
all_features = numeric_features + list(cat_features)

# Print coefficients
print("\n--- Model Coefficients ---")
print(f"Intercept: {regressor.intercept_:.2f}")
for coef, name in zip(regressor.coef_, all_features):
    print(f"{name}: {coef:.2f}")