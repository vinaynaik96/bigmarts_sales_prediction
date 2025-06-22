### Import libaries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# File paths for import
data_dir = 'data/'
train_file = data_dir + 'train_v9rqX0R.csv'
test_file = data_dir + 'test_AbJTz2l.csv'
output_file = 'ABB_submission.csv'

# 1. Load data
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# 2. Combine train and test for consistent preprocessing
test['Item_Outlet_Sales'] = np.nan
data = pd.concat([train, test], sort=False)

# 3. Standardize Item_Fat_Content using mapping
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
    'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'
})

# 4. Created Item_Type_Combined using Item_Identifier
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: {'F': 'Food', 'D': 'Drinks', 'N': 'Non-Consumable'}.get(x[0], 'Other'))

# 5. Created Item_Type_Combined 
data.loc[data['Item_Type_Combined'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'

# 6. Impute Item_Weight with group mean
data['Item_Weight'] = data['Item_Weight'].fillna(
    data.groupby('Item_Type_Combined')['Item_Weight'].transform('mean')
)

# 7. Flagged zero visibility feature replacing with NaN and impute
data['Zero_Visibility_Flag'] = (data['Item_Visibility'] == 0).astype(int)
data['Item_Visibility'] = data['Item_Visibility'].replace(0, np.nan)
data['Item_Visibility'] = data['Item_Visibility'].fillna(
    data.groupby(['Item_Type_Combined', 'Outlet_Type'])['Item_Visibility'].transform('median')
)

# 8. Impute Outlet_Size
def fill_outlet_size(row):
    mode = data[(data['Outlet_Type'] == row['Outlet_Type']) & 
                (data['Outlet_Location_Type'] == row['Outlet_Location_Type'])]['Outlet_Size'].mode()
    return mode.iloc[0] if not mode.empty else 'Medium'
data['Outlet_Size'] = data.apply(lambda row: fill_outlet_size(row) if pd.isna(row['Outlet_Size']) else row['Outlet_Size'], axis=1)

# 9. Outlet years feature
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

# 10. Log transform features for Item_MRP_log and Item_Visibility_log
data['Item_MRP_log'] = np.log1p(data['Item_MRP'])
data['Item_Visibility_log'] = np.log1p(data['Item_Visibility'])

# 11. Mean visibility ratio calculation
data['Item_Visibility_MeanRatio'] = data['Item_Visibility'] / (
    data.groupby(['Item_Identifier', 'Outlet_Type'])['Item_Visibility'].transform('mean') + 1e-8
)

# 12. Interaction features creation.
data['MRP_Outlet_Years'] = data['Item_MRP'] * data['Outlet_Years']
data['MRP_Visibility'] = data['Item_MRP'] * data['Item_Visibility']
data['MRP_Weight'] = data['Item_MRP'] * data['Item_Weight']

# 13. Category features
data['Item_Category'] = data['Item_Identifier'].str[:2]
train_m = ~data['Item_Outlet_Sales'].isna()
overall_mean = data[train_m]['Item_Outlet_Sales'].mean()

# 14. Target encoding
category_stats = data[train_m].groupby('Item_Category')['Item_Outlet_Sales'].agg(['mean', 'count'])
category_stats['smoothed'] = (category_stats['count'] * category_stats['mean'] + 10 * overall_mean) / (category_stats['count'] + 10)
data['Item_Category_TargetEncoded'] = data['Item_Category'].map(category_stats['smoothed']).fillna(overall_mean)

# 15. Ordinal encoding
sorted_means = category_stats['mean'].sort_values()
category_to_ordinal = {cat: idx for idx, cat in enumerate(sorted_means.index)}
data['Item_Category_Ordinal'] = data['Item_Category'].map(category_to_ordinal).fillna(len(category_to_ordinal))

# 16. Mean sales encoding
data['Type_Combined_MeanSales'] = data['Item_Type_Combined'].map(data[train_m].groupby('Item_Type_Combined')['Item_Outlet_Sales'].mean())
data['Outlet_MeanSales'] = data['Outlet_Identifier'].map(data[train_m].groupby('Outlet_Identifier')['Item_Outlet_Sales'].mean())
data['Category_MeanSales'] = data['Item_Category'].map(data[train_m].groupby('Item_Category')['Item_Outlet_Sales'].mean())

# 17. Label encode remaining categorical features
le_cols = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Item_Type_Combined', 'Outlet_Identifier']
for col in le_cols:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# 18. Train test spliting
train_final = data[~data['Item_Outlet_Sales'].isna()].copy()
test_final = data[data['Item_Outlet_Sales'].isna()].copy()

# 19. Final feature list after removing less relevant feature
features = [
    'Item_Weight', 'Item_Visibility', 'Item_Fat_Content', 'Item_Type_Combined',
    'Item_MRP', 'Outlet_Identifier', 'Outlet_Size',
    'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Years',
    'Item_Visibility_MeanRatio', 'Zero_Visibility_Flag',
    'Item_MRP_log', 'Item_Visibility_log',
    'MRP_Outlet_Years', 'MRP_Visibility', 'MRP_Weight',
    'Type_Combined_MeanSales', 'Outlet_MeanSales', 'Category_MeanSales',
    'Item_Category_TargetEncoded', 'Item_Category_Ordinal'
]

X = train_final[features]
y = train_final['Item_Outlet_Sales']
X_test = test_final[features]

# 20. Final imputation and scaling
imputer = KNNImputer(n_neighbors=3)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 21. Train Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 21. Hyperparameter tuning for random forest
grid_params = {'n_estimators': [400], 'max_depth': [6], 'min_samples_split': [10], 'min_samples_leaf': [2]}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid = GridSearchCV(rf, grid_params, cv=3, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# 22. Validation for data
x_preds = grid.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, x_preds))
print(f'Random Forest Validation RMSE: {rf_rmse:.2f}')

# 23. Predict final outcome
best_rf_full = RandomForestRegressor(**grid.best_params_, random_state=42, n_jobs=-1)
best_rf_full.fit(X_scaled, y)
final_preds = best_rf_full.predict(X_test_scaled)

# 24. Output submission
submission = pd.DataFrame({
    'Item_Identifier': test['Item_Identifier'].values,
    'Outlet_Identifier': test['Outlet_Identifier'].values,
    'Item_Outlet_Sales': final_preds
})
submission.to_csv(output_file, index=False)
print(f'Submission file saved as {output_file}')
