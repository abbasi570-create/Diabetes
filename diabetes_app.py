# Cell 1: Install any missing libraries (if needed; Colab usually has them)
!pip install plotly scikit-learn pandas matplotlib seaborn --quiet
# Cell 2: Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Cell 3: Load and explore the dataset
# Diabetes dataset: Predict disease progression (target) based on features like age, sex, BMI, etc.
# This is a built-in sklearn dataset, no external downloads needed, so it's error-free.
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target  # Target variable: Quantitative measure of disease progression

print("Dataset Shape:", df.shape)
print("First 5 rows:")
print(df.head())
print("\nFeature Descriptions:")
print(diabetes.DESCR[:500])  # Short description

# Cell 4: Data visualization (for "best UI" - interactive plots)
# Correlation heatmap (static, using seaborn)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Interactive scatter plot: BMI vs target
fig = px.scatter(df, x='bmi', y='target', title='BMI vs Disease Progression', 
                 labels={'bmi': 'Body Mass Index', 'target': 'Disease Progression'},
                 trendline='ols')  # Adds a trendline
fig.show()

# Cell 5: Preprocess data and split
# No major preprocessing needed, but we'll handle any NaNs (none here)
df.dropna(inplace=True)  # In case, but Diabetes has no NaNs

X = df.drop('target', axis=1)
y = df['target']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Cell 6: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Random Forest for robustness
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Cell 7: More visualizations (actual vs predicted)
# Static plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Target')
plt.ylabel('Predicted Target')
plt.title('Actual vs Predicted Disease Progression')
plt.show()

# Interactive plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Fit', line=dict(dash='dash')))
fig.update_layout(title='Actual vs Predicted (Interactive)', xaxis_title='Actual Target', yaxis_title='Predicted Target')
fig.show()

# Cell 8: Feature importance (useful for UI in Streamlit)
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.show()

# Cell 9: Save the model for Streamlit
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as 'model.pkl'. Download this file for your Streamlit app.")

# Cell 10: Example prediction function (for Streamlit integration)
def predict_progression(features):
    """
    Predict disease progression based on input features.
    Features should be a list or array in the order: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    """
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Test the function
sample_features = [0.038076, 0.050680, 0.061696, 0.021872, -0.044223, -0.034821, -0.043401, -0.002592, 0.019907, -0.017646]  # From dataset
predicted_progression = predict_progression(sample_features)
print(f"Predicted progression for sample features: {predicted_progression:.2f}")

# Cell 11: %%writefile to save this as a .py file (for easy download and Streamlit use)

# This saves the entire notebook code as a .py file. You can download it from Colab's file explorer.
# In Streamlit, you can load 'model.pkl' and use the predict_progression function.

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Preprocess and train (same as above)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

def predict_progression(features):
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]

# Example usage in Streamlit: Load model and use predict_progression
# In your Streamlit app, add: model = pickle.load(open('model.pkl', 'rb'))
