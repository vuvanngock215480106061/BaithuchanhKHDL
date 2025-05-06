import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Đọc dữ liệu
df = pd.read_csv('Data_Number_7.csv')

# 1. Tạo chỉ số nguy cơ biến chứng
df['risk_score'] = df['bmi'] * df['blood_glucose'] + df['hospitalizations']

# 2. Kiểm định Chi-squared với nhóm tuổi
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 100], labels=['<40', '40-60', '>60'])
contingency_table = pd.crosstab(df['age_group'], df['complication'])
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-squared test p-value: {p:.4f}")

# 3. Tạo đặc trưng xu hướng đường huyết (giả định)
df['glucose_trend'] = np.where(df['blood_glucose'] > 150, 'increase',
                              np.where(df['blood_glucose'] < 80, 'decrease', 'stable'))

# 4. Tạo đặc trưng mức độ nghiêm trọng
df['severity'] = df['hospitalizations'] * df['blood_glucose']

# 5. Chuẩn bị dữ liệu cho mô hình
X = df[['bmi', 'blood_glucose', 'hospitalizations', 'severity']]
y = df['complication']

# Xử lý mất cân bằng bằng SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Chia tập train-test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 6. Huấn luyện Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print("Logistic Regression Performance:")
print(classification_report(y_test, lr.predict(X_test)))

# 7. Tối ưu Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Random Forest Parameters:", grid_search.best_params_)
print("Random Forest Performance:")
print(classification_report(y_test, grid_search.predict(X_test)))