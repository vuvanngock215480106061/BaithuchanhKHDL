import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
df = pd.read_csv('Data_Number_5.csv')

# 1. Tạo chỉ số hiệu suất
df['performance'] = df[['math_score', 'literature_score', 'science_score']].mean(axis=1) * df['study_hours']

# Kiểm tra giá trị bị thiếu (nếu có) và thay thế hoặc loại bỏ
df = df.dropna(subset=['math_score', 'literature_score', 'science_score', 'study_hours', 'extracurricular'])

# 2. Kiểm định ANOVA
high = df[df['extracurricular'] == 'High']['performance']
medium = df[df['extracurricular'] == 'Medium']['performance']
low = df[df['extracurricular'] == 'Low']['performance']
f_stat, p_val = f_oneway(high, medium, low)
print(f"ANOVA p-value: {p_val:.4f}")

# 3. Tạo đặc trưng cân bằng học tập
df['balanced'] = df[['math_score', 'literature_score', 'science_score']].std(axis=1)

# 4. Tạo đặc trưng rủi ro học tập
df['risk'] = np.where((df['absences'] > 5) & (df['study_hours'] < 10), 1, 0)

# 5. Chuẩn bị dữ liệu cho SVM
X = df[['performance', 'balanced', 'risk']]
y = np.where(df['math_score'] < 50, 1, 0)  # Giả định trượt nếu điểm toán <50

# Chia tập train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Tối ưu SVM
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
print("Best SVM Parameters:", grid_search.best_params_)

# Dự đoán và in kết quả
y_pred = grid_search.predict(X_test_scaled)
print("SVM Performance:")
print(classification_report(y_test, y_pred, zero_division=1))
