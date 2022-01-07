#=================== PREDICTING CREDIT CARD FRAUD ====================

#==== 0. DEPENDENCIES 
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

sns.set_theme(palette="viridis")
warnings.filterwarnings('ignore') 

#==== 1. DATA AND PREPROCESSING
# IMPORTING
data = pd.read_csv("project_1/data.csv")    #Data with credit card info (N=80,000 excluding repeated obs.)

# PREPROCESSING
data.drop_duplicates(inplace=True)
data = data.sample(n=20000, random_state=123)      #Sample: n=20,000 for faster run-time (≈1min) 
data.reset_index(inplace=True, drop=True)

    # Creating features and label
feature_cols = [x for x in data.columns if x not in 'Class']
X = data.loc[:, feature_cols]
y = data.loc[:, 'Class']


#     # Plotting abs correlation
# correlations = data[feature_cols].corrwith(y)
# abs_correlations = pd.DataFrame(zip(correlations.index, abs(correlations.values)))
# abs_correlations.sort_values(by=1 ,inplace=True, ascending=False)
# abs_correlations.plot.bar(x=0, xlabel='', ylabel='Absolute Correlation with Target', legend=False)
# plt.show()


# TRAIN/TEST SPLITS
    # Splitting the data into two parts with 30% in the test data (for cross-validation)
strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=int(3*data.shape[0]/10), random_state=123) #This creates a generator
train_idx, test_idx = next(strat_shuff_split.split(data[feature_cols], data['Class'])) #Get the index values from the generator

    # Creating the data sets
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'Class']

X_test = data.loc[test_idx, feature_cols]
y_test = data.loc[test_idx, 'Class']

#==== 2. TRAINING
# Function that returns metrics for validation
def measure_error(y_true, y_pred, label):
    return pd.Series({'roc_auc':roc_auc_score(y_true, y_pred),
                      'precision': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'r2': r2_score(y_true, y_pred)},
                      name=label)
models = [None] * 4         #List of tuned models

# MODEL 1 (models[0]): Logistic Regression
    # Hyperparameter tuning
h_params = np.geomspace(1e-4, 1e0, 10) #Regularization parameter to avoid overfitting

scores = []
estimators = []
for C in h_params:
    estimator = Pipeline([("scaler", StandardScaler()), #Scaling the features
        ("LR", LogisticRegression(max_iter=1e8, C=C, random_state=123, n_jobs=-1))])

    estimator.fit(X_train, y_train)
    y_hat = estimator.predict(X_test)
    score =  measure_error(y_test, y_hat, 'Model 1').roc_auc    # ROC AUC is generally preferred to other measurements for cross-validation
    scores.append(score)
    estimators.append(estimator)
idx = scores.index(max(scores))
models[0] = estimators[idx]     #Best estimator

    # Plotting
# sns.lineplot(x=h_params, y=scores
#             ,marker="o")

# ax = plt.gca()
# ax.set(xlabel='C', ylabel='ROC-AUC');
# plt.show()

# MODEL 2 (models[1]): Support Vector Machine (SVM)
    # Hyperparameter tuning
h_params = np.geomspace(1e-1, 1e1, 10)

scores = []
estimators = []
for C in h_params:
    estimator = Pipeline([("scaler", StandardScaler()),
        ("SVC", SVC(kernel='linear', max_iter=1e7, C=C, random_state=123, probability=True))])

    estimator.fit(X_train, y_train)
    y_hat = estimator.predict(X_test)
    score =  measure_error(y_test, y_hat, 'Model 2').roc_auc
    scores.append(score)
    estimators.append(estimator)
idx = scores.index(max(scores))
models[1] = estimators[idx]     #Best estimator

    # Plotting
# sns.lineplot(x=h_params, y=scores
#             ,marker="o")

# ax = plt.gca()
# ax.set(xlabel='C', ylabel='ROC-AUC');
# plt.show()

# MODEL 3 (models[2]): Random Forest
    # Hyperparameter tuning
h_params = [10, 20, 50, 100, 150, 200, 500]

scores = []
estimators = []
for n_est in h_params:
    estimator = Pipeline([("scaler", StandardScaler()),
        ("RandomForest", RandomForestClassifier(n_estimators=n_est, max_features=8, random_state=123, n_jobs=-1))])

    estimator.fit(X_train, y_train)
    y_hat = estimator.predict(X_test)
    score =  measure_error(y_test, y_hat, 'Model 3').roc_auc
    scores.append(score)
    estimators.append(estimator)
idx = scores.index(max(scores))
models[2] = estimators[idx]     #Best estimator

    # Plotting
# sns.lineplot(x=h_params, y=scores
#             ,marker="o")

# ax = plt.gca()
# ax.set(xlabel='C', ylabel='ROC-AUC');
# plt.show()

# MODEL 4 (models[3]): Voting Classifier (combination of the previous three models)
estimators = [('Logistic Regression', models[0]),
            ('SVM', models[1]),
            ('Random Forest', models[2])]
models[3] = VotingClassifier(estimators, voting='soft', n_jobs=-1)
models[3].fit(X_train, y_train)

#==== 3. RESULTS
# COMPUTING METRICS
y_hat = [None] * 4
aux = [None] * 4

for i in range(4):
    y_hat[i] = models[i].predict(X_test)
    aux[i] = measure_error(y_test, y_hat[i], 'Model '+str(i))

results = pd.concat([aux[0], aux[1], aux[2], aux[3]],
                              axis=1)
results.columns = ['Logistic Reg.', 'SVM', 'Rand. Forest', 'Voting Classifier']
print(results)

# PLOTTING
plot = results.unstack().reset_index() 
plot.columns = ["Model", "Metric", "Value"]
ax = sns.barplot(x="Model", y="Value", hue="Metric", data=plot)
ax.set(ylabel="")
plt.show()

r2s = list(results.iloc[3])
idx = r2s.index(max(r2s))
best_model = results.columns[idx]
print (f"\nBest model: {best_model}\nR-squared: {max(r2s):.4f}")