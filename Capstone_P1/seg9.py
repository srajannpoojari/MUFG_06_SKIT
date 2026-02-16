# ========================================
# SEGMENT 9: SAVE MODEL & PREPROCESSING COMPONENTS
# ========================================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

print("="*60)
print("SAVING MODEL & PREPROCESSING COMPONENTS")
print("="*60)

# ========== 1. LOAD DATA ==========
df_raw = pd.read_csv('manufacturing_dataset_cleaned.csv')
y = df_raw['Parts_Per_Hour']
X = df_raw.drop('Parts_Per_Hour', axis=1)

# ========== 2. ONE-HOT ENCODE ==========
X_encoded = pd.get_dummies(X, columns=['Shift', 'Machine_Type', 'Material_Grade', 'Day_of_Week'], drop_first=False)

# ========== 3. TRAIN-TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# ========== 4. FIT SCALER ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ========== 5. TRAIN MODEL ==========
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ========== 6. SAVE MODEL - PICKLE ONLY ==========
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Saved: model.pkl")

# ========== 7. SAVE SCALER - PICKLE ONLY ==========
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: scaler.pkl")

# ========== 8. SAVE FEATURE COLUMNS ==========
feature_columns = X_encoded.columns.tolist()
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print(f"✓ Saved: feature_columns.pkl ({len(feature_columns)} features)")

# ========== 9. SAVE CATEGORICAL MAPPINGS ==========
categorical_mappings = {
    'Shift': df_raw['Shift'].unique().tolist(),
    'Machine_Type': df_raw['Machine_Type'].unique().tolist(),
    'Material_Grade': df_raw['Material_Grade'].unique().tolist(),
    'Day_of_Week': df_raw['Day_of_Week'].unique().tolist()
}
with open('categorical_mappings.pkl', 'wb') as f:
    pickle.dump(categorical_mappings, f)
print("✓ Saved: categorical_mappings.pkl")

print("\n" + "="*60)
print("✅ ALL FILES SAVED WITH PICKLE (NO JOBLIB)")
print("   - model.pkl")
print("   - scaler.pkl")
print("   - feature_columns.pkl")
print("   - categorical_mappings.pkl")
print("="*60)