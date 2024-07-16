import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp

def plot_columns(df, column1, column2):
    """
    Funkcia na vykreslenie grafu dvoch zvolených stĺpcov z DataFrame.
    
    :param df: pandas DataFrame obsahujúci dáta
    :param column1: názov prvého stĺpca
    :param column2: názov druhého stĺpca
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df[column1], df[column2], marker='o', linestyle='--')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title(f'Graph {column1} vs {column2}')
    plt.grid(True)
    plt.show()

def distribution(df, column):
    """
    Funkcia na vykreslenie distribúcie zvoleného stĺpca.
    
    :param df: pandas DataFrame obsahujúci dáta
    :param column: názov stĺpca, ktorého distribúciu chceme vykresliť
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'{column} Distribution')
    plt.grid(True)
    plt.show()

df = pd.read_csv('logreg_data.csv',delimiter = ';')

# First analysis
#df.describe().to_csv('statistics.csv', sep=';', index=True)
unique_counts = df.nunique()
value_counts = {col: df[col].value_counts() for col in df.columns}
nan_counts = df.isna().sum()

# ---------------------------------------------------------------------------------------------------------
# Task I. How looks the distribution of age of a customer? 
# Correcting the age in the dataset
this_year = datetime.today().year
df['Age'] = df['Year of birth'].apply(lambda x: this_year - x if x <= this_year else this_year - x + 100)
#df['Age'] = df['Year of birth'].apply(lambda x: this_year - x if x <= this_year else this_year - x)
distribution(df,"Age")

# Task I. And how default rate depends on the age? 
age_counts = df.groupby('Age').size()
default_counts = df.groupby('Age')['DEFAULT'].sum() 
df_default = pd.DataFrame({
	'Age': age_counts.index,
    'AgeCount': age_counts.values,
    'DefaultCounts': default_counts.values,
    'DefaultRate': (default_counts / age_counts) * 100
}).reset_index(drop=True)
plot_columns(df_default,'Age','DefaultRate')

# ---------------------------------------------------------------------------------------------------------
# Task II. Which variables are the main drivers of default? 
# Firstly, we need to make dummy variables from categoric ones.
categoric_vars = ['City', 'Gender', 'Education']
string_columns = df.select_dtypes(include=['object']).columns
df[string_columns] = df[string_columns].fillna('Unknown')
df["Loan Amount"] = df["Loan Amount"].fillna(0)
df_encoded = pd.get_dummies(df, columns=categoric_vars)
#df_encoded.to_csv('data.csv', sep=';', index=False)

# Correlation matrix
corr_matrix = df_encoded.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f', linewidths=.5)
plt.show()

# Using logaritmic regression to see which attributes are important
X = df_encoded.drop('DEFAULT', axis=1)
y = df_encoded['DEFAULT']
X = X.drop(columns=['Gender_Female','City_Prague', 'Education_Primary','Age'])

X = sm.add_constant(X)
for col in X.select_dtypes(include=[bool]).columns:
    X[col] = X[col].astype(int)

# Check for multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

logit_model = sm.Logit(y, X)
result = logit_model.fit()
summary_df = result.summary2().tables[1]

# Explanatory variables with p-value < 0.05
significant_vars = summary_df[summary_df['P>|z|'] < 0.05]
EVs = significant_vars.index.tolist()
print(significant_vars[['Coef.','P>|z|']])

# -------------------------------------------------------------------------------------------------------
# Task III. Design and implement a predictive model for customer probability of default using logistic regression 
X = X[EVs]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]  # Probability of DEFAULT = 1

# III. (bonus task: predict also the default event itself).
y_pred = model.predict(X_test)


# ---------------------------------------------------------------------------------------------------------
# Task V. Bonus question: Calculate AUC/GINI or Kolmogorov-Smirnov performance measures for the obtained model.
auc = roc_auc_score(y_test, y_proba)
gini = 2 * auc - 1
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
ks_stat = max(tpr - fpr)

print(f"AUC: {auc:.4f}")
print(f"GINI: {gini:.4f}")
print(f"KS Statistic: {ks_stat:.4f}")

# ------------------------------------------------------------------------------------------------------------------------------
# Coefficients for Excel Output
coefficients = model.coef_[0]  
intercept = model.intercept_[0] 
features = X.columns 

coef_df = pd.DataFrame({
    'Feature': features.tolist() + ['Intercept'],
    'Coefficient': list(coefficients) + [intercept]
})

#coef_df.to_csv("coef.csv", sep = ';')


#------------------------------------------------------------------------------------------------------------------------------
# Prediction of probability of DEFAULT for input data
input_data = [
    {'Loan term': 36, 'Loan Amount': 2384, 'Year of application': 2012, 'Year of birth': 1979, 'City': 'Prague', 'Gender': 'Female', 'Education': 'Primary'},
    {'Loan term': 12, 'Loan Amount': 3499, 'Year of application': 2015, 'Year of birth': 1986, 'City': 'Brno', 'Gender': 'Female', 'Education': 'Primary'}
]

#-------------------------------------------------------------------------------------------------------------------------------
df_input = pd.DataFrame(input_data)
string_columns = df_input.select_dtypes(include=['object']).columns
df_input[string_columns] = df_input[string_columns].fillna('Unknown')
df_input["Loan Amount"] = df_input["Loan Amount"].fillna(0)

# One-Hot Encoding
df_input_encoded = pd.get_dummies(df_input, columns= categoric_vars)

train_columns = X.columns.tolist()
test_columns = df_input_encoded.columns.tolist()
missing_cols = [col for col in train_columns if col not in test_columns]
for col in missing_cols:
    df_input_encoded[col] = False

df_input_encoded = df_input_encoded[train_columns]

y_proba = model.predict_proba(df_input_encoded)[:, 1]  # Pravdepodobnosť pre triedu 1 (zlyhanie)

print("Predicted probability of DEFAULT for input data: ",y_proba)