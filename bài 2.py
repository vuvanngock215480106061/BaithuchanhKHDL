import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest, StackingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import optuna
import shap
import warnings
warnings.filterwarnings("ignore")

# 1. Tải dữ liệu Boston Housing
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame

# Đổi tên cột MEDV thành PRICE
df.rename(columns={'MEDV': 'PRICE'}, inplace=True)
df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')

# 2. Vẽ pairplot
sns.pairplot(df[['CRIM', 'RM', 'LSTAT', 'PRICE']])
plt.show()

# 3. Pearson correlation
corr = df.corr()
print("Top features correlated with PRICE:")
print(corr['PRICE'].sort_values(ascending=False).head(5))

# 4. Xử lý outliers bằng Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(df)
df = df[outliers == 1]

# 5. Kiểm tra VIF
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

def calculate_vif(data):
    data = data.select_dtypes(include=[np.number]).astype(float)  # chỉ giữ cột số và ép kiểu float
    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

# Khởi tạo dữ liệu đầu vào VIF
X_vif = df.drop('PRICE', axis=1).select_dtypes(include=[np.number]).astype(float)
vif = calculate_vif(X_vif)

# Lặp lại nếu VIF > 5
while vif['VIF'].max() > 5:
    remove_feature = vif.loc[vif['VIF'].idxmax(), 'feature']
    print(f"Removing '{remove_feature}' due to high VIF = {vif['VIF'].max():.2f}")
    df.drop(remove_feature, axis=1, inplace=True)
    X_vif = df.drop('PRICE', axis=1).select_dtypes(include=[np.number]).astype(float)
    vif = calculate_vif(X_vif)


# 6. Tạo đặc trưng mới
if 'RM' in df.columns and 'CRIM' in df.columns:
    df['room_per_crime'] = df['RM'] / (df['CRIM'] + 1e-6)

if 'TAX' in df.columns:
    df['high_tax'] = (df['TAX'] > df['TAX'].mean()).astype(int)

if 'RM' in df.columns and 'LSTAT' in df.columns:
    df['RM_LSTAT'] = df['RM'] * df['LSTAT']


# 7. Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df.drop('PRICE', axis=1))
df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out())
X = df_poly
y = df['PRICE']

# 8. Tách dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Tiền xử lý
preprocessor = ColumnTransformer(transformers=[('scaler', StandardScaler(), X.columns)])

# 10. Mô hình
models = {
    'Linear Regression': LinearRegression(),
    'XGBoost': XGBRegressor(random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
}

# 11. Tối ưu XGBoost với Optuna
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10)
    }
    model = XGBRegressor(**params, random_state=42)
    return -np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_xgb = XGBRegressor(**study.best_params, random_state=42)

# 12. Stacking Regressor
estimators = [
    ('lr', LinearRegression()),
    ('xgb', best_xgb),
    ('nn', MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42))
]
stacking = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# 13. Đánh giá cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(model_pipeline, X, y, cv=kf, scoring='r2')
    print(f"{name} R²: {np.mean(scores):.4f}")

# 14. SHAP Analysis
best_xgb.fit(X_train, y_train)
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
