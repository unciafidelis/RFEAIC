import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
import pandas as pd

# Load the dataset
datos = pd.read_csv('adnimerge-EObsNA_2_SanovsEnfermo.csv')  
df = datos 
print('adnimerge-EObsNA_2_SanovsEnfermo')

X = df.drop(['DX.bl'], axis=1)
y = df['DX.bl'].astype(float)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Standardize the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize variables to store the best results
best_aic_lr = float('inf')
best_selected_features_lr = []
best_accuracy_lr = 0
best_model_lr = None

best_aic_svm = float('inf')
best_selected_features_svm = []
best_accuracy_svm = 0
best_model_svm = None

best_aic_rf = float('inf')
best_selected_features_rf = []
best_accuracy_rf = 0
best_model_rf = None

# Feature Selector
for num_features in range(len(X.columns), 3, -1):  # decreasing order from len(all_features) to 4
    
    # Logistic Regression
    estimatorLR = LogisticRegression(max_iter=1500)
    selector_lr = RFECV(estimatorLR, cv=10)
    selector_lr = selector_lr.fit(X, y)
    all_features_lr = X.columns
    selected_features_lr = [all_features_lr[i] for i in range(len(all_features_lr)) if selector_lr.support_[i]]
    
    # AIC calculation for Logistic Regression
    n = len(y)
    k_lr = len(selected_features_lr) + 1  # add 1 for the intercept term
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train[:, selector_lr.support_], y_train)
    ll_lr = lr_model.score(X_train[:, selector_lr.support_], y_train)
    aic_lr = 2 * k_lr - 2 * ll_lr
    
    # Accuracy calculation for Logistic Regression
    accuracy_lr = lr_model.score(X_test[:, selector_lr.support_], y_test)
    
    # Update the best results for Logistic Regression
    if aic_lr < best_aic_lr and accuracy_lr > best_accuracy_lr:
        best_aic_lr = aic_lr
        best_selected_features_lr = selected_features_lr
        best_accuracy_lr = accuracy_lr
        best_model_lr = lr_model
    
    # Support Vector Machine (SVM) kernel = 'linear' 'rbf'
    estimatorSVM = SVR(kernel='linear')
    selector_svm = RFECV(estimatorSVM, cv=10)
    selector_svm = selector_svm.fit(X, y)
    all_features_svm = X.columns
    selected_features_svm = [all_features_svm[i] for i in range(len(all_features_svm)) if selector_svm.support_[i]]
    
    # AIC calculation for SVM
    n = len(y)
    k_svm = len(selected_features_svm) + 1  # add 1 for the intercept term
    svm_model = SVR(kernel='linear')
    svm_model.fit(X_train[:, selector_svm.support_], y_train)
    ll_svm = svm_model.score(X_train[:, selector_svm.support_], y_train)
    aic_svm = 2 * k_svm - 2 * ll_svm
    
    # Accuracy calculation for SVM
    accuracy_svm = svm_model.score(X_test[:, selector_svm.support_], y_test)
    
    # Update the best results for SVM
    if aic_svm < best_aic_svm and accuracy_svm > best_accuracy_svm:
        best_aic_svm = aic_svm
        best_selected_features_svm = selected_features_svm
        best_accuracy_svm = accuracy_svm
        best_model_svm = svm_model
    
    # Random Forest (RF)
    estimatorRF = RandomForestClassifier(n_estimators=100, random_state=5)
    selector_rf = RFECV(estimatorRF, cv=10)
    selector_rf = selector_rf.fit(X, y)
    all_features_rf = X.columns
    selected_features_rf = [all_features_rf[i] for i in range(len(all_features_rf)) if selector_rf.support_[i]]
    
    # AIC calculation for Random Forest
    n = len(y)
    k_rf = len(selected_features_rf) + 1  # add 1 for the intercept term
    rf_model = RandomForestClassifier(n_estimators=100, random_state=5)
    rf_model.fit(X_train[:, selector_rf.support_], y_train)
    ll_rf = rf_model.score(X_train[:, selector_rf.support_], y_train)
    aic_rf = 2 * k_rf - 2 * ll_rf
    
    # Accuracy calculation for Random Forest
    accuracy_rf = rf_model.score(X_test[:, selector_rf.support_], y_test)
    
    # Update the best results for Random Forest
    if aic_rf < best_aic_rf and accuracy_rf > best_accuracy_rf:
        best_aic_rf = aic_rf
        best_selected_features_rf = selected_features_rf
        best_accuracy_rf = accuracy_rf
        best_model_rf = rf_model

# Print the best results for each model
print("LR Best Selected Features: ", best_selected_features_lr)
print("LR Best AIC: ", best_aic_lr)
print("LR Best Accuracy: ", best_accuracy_lr)

print("SVM Best Selected Features: ", best_selected_features_svm)
print("SVM Best AIC: ", best_aic_svm)
print("SVM Best Accuracy: ", best_accuracy_svm)

print("RF Best Selected Features: ", best_selected_features_rf)
print("RF Best AIC: ", best_aic_rf)
print("RF Best Accuracy: ", best_accuracy_rf)
