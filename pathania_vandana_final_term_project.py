import pandas as pd
import numpy as np
import random
import tensorflow as tf
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss

from keras.models import Sequential
from keras.layers import Dense, LSTM

# Set random seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Load and preprocess the dataset
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    print("Dataset Information:")
    print(data.info())
    
    # Encode categorical variables
    categorical_columns = ['type_school', 'school_accreditation', 'gender', 'interest', 'residence', 'parent_was_in_college', 'will_go_to_college']
    label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_columns}
    for col, le in label_encoders.items():
        data[col] = le.transform(data[col])
    
    return data

data = load_and_preprocess_data('data.csv')

# Split the data into features and labels
features = data.drop(['will_go_to_college'], axis=1)
labels = data['will_go_to_college']

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)
feature_columns = data.drop(['will_go_to_college'], axis=1).columns
features_df = pd.DataFrame(features, columns=feature_columns)
print(features_df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42, stratify=labels)

# Define Stratified K-Fold cross-validator
cv_stratified = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define model parameters
knn_params = {"n_neighbors": list(range(3, 20, 2))}
rf_params = {"n_estimators": list(range(50, 201, 25)), "min_samples_split": [5, 10, 15, 20]}

# Perform GridSearchCV for KNN and Random Forest
def perform_grid_search(model, params, X_train, y_train):
    cv = GridSearchCV(model, params, cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)
    return cv.best_params_

best_knn_params = perform_grid_search(KNeighborsClassifier(), knn_params, X_train, y_train)
best_rf_params = perform_grid_search(RandomForestClassifier(), rf_params, X_train, y_train)

# Function to calculate metrics
def calc_metrics(y_true, y_pred, y_prob):
    # Confusion matrix components
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Helper function to avoid division by zero
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0

    # Calculate metrics
    metrics = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TPR': safe_divide(TP, TP + FN),  # True Positive Rate
        'TNR': safe_divide(TN, TN + FP),  # True Negative Rate
        'FPR': safe_divide(FP, TN + FP),  # False Positive Rate
        'FNR': safe_divide(FN, TP + FN),  # False Negative Rate
        'Precision': safe_divide(TP, TP + FP),  # Precision
        'F1_measure': safe_divide(2 * TP, 2 * TP + FP + FN),  # F1 Measure
        'Accuracy': safe_divide(TP + TN, TP + FP + FN + TN),  # Accuracy
        'Error_rate': 1 - safe_divide(TP + TN, TP + FP + FN + TN),  # Error Rate
        'BACC': safe_divide(safe_divide(TP, TP + FN) + safe_divide(TN, TN + FP), 2),  # Balanced Accuracy
        'TSS': safe_divide(TP, TP + FN) - safe_divide(FP, TN + FP),  # True Skill Statistic
        'HSS': safe_divide(2 * (TP * TN - FP * FN), ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))),  # Heidke Skill Score
        'Brier_score': brier_score_loss(y_true, y_prob),  # Brier Score
        'AUC': roc_auc_score(y_true, y_prob)  # Area Under the Curve
    }

    # Return metrics as a list in a specific order
    return [metrics[key] for key in ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 'Precision', 'F1_measure', 'Accuracy', 'Error_rate', 'BACC', 'TSS', 'HSS', 'Brier_score', 'AUC']]

# Function to get metrics for models
def get_metrics(model, X_train, X_test, y_train, y_test, lstm_flag=False):
    if lstm_flag:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    return calc_metrics(y_test, y_pred, y_prob)

# LSTM model definition
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(input_shape, 1), return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

lstm_model = create_lstm_model(X_train.shape[1])

# Initialize metric columns and lists
metric_columns = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 'Precision', 'F1_measure', 'Accuracy', 'Error_rate', 'BACC', 'TSS', 'HSS', 'Brier_score', 'AUC']
knn_metrics_list, rf_metrics_list, lstm_metrics_list = [], [], []

# 10 Iterations of 10-fold cross-validation
for iter_num, (train_index, test_index) in enumerate(cv_stratified.split(X_train, y_train), start=1):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index].values, y_train.iloc[test_index].values
    
    # KNN Model
    knn_model = KNeighborsClassifier(n_neighbors=best_knn_params['n_neighbors'])
    # Random Forest Model
    rf_model = RandomForestClassifier(min_samples_split=best_rf_params['min_samples_split'], n_estimators=best_rf_params['n_estimators'])
    
    # Get metrics for each algorithm
    knn_metrics = get_metrics(knn_model, X_train_fold, X_test_fold, y_train_fold, y_test_fold, lstm_flag=False)
    rf_metrics = get_metrics(rf_model, X_train_fold, X_test_fold, y_train_fold, y_test_fold, lstm_flag=False)
    lstm_metrics = get_metrics(lstm_model, X_train_fold, X_test_fold, y_train_fold, y_test_fold, lstm_flag=True)
    
    # Append metrics to respective lists
    knn_metrics_list.append(knn_metrics)
    rf_metrics_list.append(rf_metrics)
    lstm_metrics_list.append(lstm_metrics)
    
    # Create a DataFrame for all metrics in the current iteration
    metrics_all_df = pd.DataFrame([knn_metrics, rf_metrics, lstm_metrics],
                                  columns=metric_columns, index=['KNN', 'RF', 'LSTM'])

# Create a DataFrame for the metrics of each algorithm
metric_index_df = ['iter' + str(i) for i in range(1, 11)]
knn_metrics_df = pd.DataFrame(knn_metrics_list, columns=metric_columns, index=metric_index_df)
rf_metrics_df = pd.DataFrame(rf_metrics_list, columns=metric_columns, index=metric_index_df)
lstm_metrics_df = pd.DataFrame(lstm_metrics_list, columns=metric_columns, index=metric_index_df)

# Display metrics for each algorithm in each iteration
for algo_name, metrics_df in zip(['KNN', 'RF', 'LSTM'], [knn_metrics_df, rf_metrics_df, lstm_metrics_df]):
    print('\nMetrics for Algorithm {}:\n'.format(algo_name))
    print(metrics_df.round(decimals=2).T)
    print('\n')

# Calculate the average metrics for each algorithm
knn_avg_df = knn_metrics_df.mean()
rf_avg_df = rf_metrics_df.mean()
lstm_avg_df = lstm_metrics_df.mean()

# Create a DataFrame with the average performance for each algorithm
avg_performance_df = pd.DataFrame({'KNN': knn_avg_df, 'RF': rf_avg_df, 'LSTM': lstm_avg_df}, index=metric_columns)

# Display the average performance for each algorithm
print('\n----- Average Performance for Each Algorithm -----\n')
print(avg_performance_df.round(decimals=2))
