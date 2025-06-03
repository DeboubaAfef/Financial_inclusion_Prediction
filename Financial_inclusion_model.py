import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import joblib

# 🔹 Load the dataset
df = pd.read_csv("Financial_inclusion_dataset.csv")
print(df.shape)
df.info()
print(df.head())

# 🔹 Generate profiling report
profile = ProfileReport(df, title="Financial Inclusion Profiling Report", explorative=True)
profile.to_file("Financial_inclusion_report.html")
print("✔ Report generated: 'Financial_inclusion_report.html'")

# 🔹 Clean the data
df_cleaned = df.copy()
print(df_cleaned.isnull().sum())  # Missing values
print(df_cleaned.duplicated().sum())  # Duplicates
df_cleaned = df_cleaned.drop_duplicates()

# 🔹 Handle outliers
numeric_columns = ['household_size', 'age_of_respondent', 'year']
plt.figure(figsize=(15, 5))
for i, column in enumerate(numeric_columns):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x=df_cleaned[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()
plt.savefig("All_Outliers_Boxplots.png")
print("✔ Boxplots saved as 'All_Outliers_Boxplots.png'")

# Function to remove outliers
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    original_count = data.shape[0]
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed = original_count - data.shape[0]
    print(f"✔ Outliers removed from '{column}': {removed} rows")
    return data

df_cleaned = remove_outliers_iqr(df_cleaned, 'household_size')
df_cleaned = remove_outliers_iqr(df_cleaned, 'age_of_respondent')

# 🔹 Encode categorical columns
categorical_cols = [
    'country', 'location_type', 'cellphone_access', 'gender_of_respondent',
    'relationship_with_head', 'marital_status', 'education_level', 'job_type', 'bank_account'
]
le = LabelEncoder()
for col in categorical_cols:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    print(f"✔ Encoded '{col}'")

print("\n🎯 Sample of Encoded Data:")
print(df_cleaned[categorical_cols].head())

# 🔹 Prepare for ML
y = df_cleaned['bank_account']
X = df_cleaned.drop(['bank_account', 'uniqueid'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Train XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
print("✅ XGBoost model trained successfully!")

# 🔹 Predictions & Evaluation
y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n📊 Accuracy:  {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"📈 Recall:    {recall:.4f}")
print(f"🏅 F1 Score:  {f1:.4f}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Account", "Has Account"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# 🔹 Save the model
joblib.dump(xgb_model, "xgboost_model.pkl")
print("✔ Model saved as 'xgboost_model.pkl'")

print(y.value_counts(normalize=True))
