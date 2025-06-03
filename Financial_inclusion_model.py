# 🔹 Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier, plot_importance
import joblib

# 🔹 Load the dataset
df = pd.read_csv("Financial_inclusion_dataset.csv")
print("✅ Dataset loaded:", df.shape)

# 🔹 Data cleaning
df_cleaned = df.copy()
print("\n🧼 Missing values:\n", df_cleaned.isnull().sum())
print("📌 Duplicates:", df_cleaned.duplicated().sum())
df_cleaned = df_cleaned.drop_duplicates()
# 🔹 Visualize outliers
numeric_columns = ['household_size', 'age_of_respondent', 'year']
plt.figure(figsize=(15, 5))
for i, column in enumerate(numeric_columns):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x=df_cleaned[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()
plt.savefig("All_Outliers_Boxplots.png")
print("📸 Boxplots saved as 'All_Outliers_Boxplots.png'")

# 🔹 Outlier removal function
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    original_count = data.shape[0]
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed = original_count - data.shape[0]
    print(f"🗑️ Removed {removed} outliers from '{column}'")
    return data
# 🔹 Remove outliers
df_cleaned = remove_outliers_iqr(df_cleaned, 'household_size')
df_cleaned = remove_outliers_iqr(df_cleaned, 'age_of_respondent')
# 🔹 Encode categorical variables
categorical_cols = [
    'country', 'location_type', 'cellphone_access', 'gender_of_respondent',
    'relationship_with_head', 'marital_status', 'education_level', 'job_type', 'bank_account'
]
le = LabelEncoder()
for col in categorical_cols:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    print(f"🔤 Encoded '{col}'")
# 🔹 Feature selection
y = df_cleaned['bank_account']
X = df_cleaned.drop(['bank_account', 'uniqueid'], axis=1)

# 🔹 Train-test split
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
print("✅ XGBoost model trained!")

# 🔹 Evaluation
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n🎯 Evaluation Metrics:")
print(f"📊 Accuracy :  {accuracy:.4f}")
print(f"🏅 Precision:  {precision:.4f}")
print(f"📈 Recall   :  {recall:.4f}")
print(f"⭐ F1 Score :  {f1:.4f}")
# 🔹 Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Account", "Has Account"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
# 🔹 Feature importance visualization
plt.figure(figsize=(10, 6))
plot_importance(xgb_model, importance_type='gain', max_num_features=10, title='Top 10 Feature Importances')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("📊 Feature importance saved as 'feature_importance.png'")

# 🔹 Class distribution
print("\n📌 Target Class Distribution (normalized):")
print(y.value_counts(normalize=True))
