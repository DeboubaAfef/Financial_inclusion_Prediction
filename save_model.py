import joblib
from xgboost import XGBClassifier
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("Financial_inclusion_dataset.csv")

# Minimal cleaning for saving test (adjust based on your previous steps)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Label encode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in [
    'country', 'location_type', 'cellphone_access', 'gender_of_respondent',
    'relationship_with_head', 'marital_status', 'education_level', 'job_type', 'bank_account'
]:
    df[col] = le.fit_transform(df[col])

X = df.drop(['bank_account', 'uniqueid'], axis=1)
y = df['bank_account']

from sklearn.model_selection import train_test_split
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "xgboost_model.pkl")
print("✅ Model saved successfully as 'xgboost_model.pkl'")
