import pandas as pd
import os
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from colorama import Fore, Style, init

# Add parent directory to path to import backend modules if needed, roughly just for finding data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

init(autoreset=True)

# Path to data
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/archive/Lead Scoring.csv'))

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def get_preprocessor():
    numeric_features = ['Total Time Spent on Website']
    categorical_features = ['Lead Origin', 'Lead Source', 'Last Activity', 'Tags']
    
    numeric_transformer = SimpleImputer(strategy='median')
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def evaluate_model(name, clf, X_train, y_train, X_test, y_test):
    print(f"\n{Fore.CYAN}Training {name}...{Style.RESET_ALL}")
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Inference
    start_time = time.time()
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    inference_time = (time.time() - start_time) / len(X_test) # per sample
    
    
    # Custom Metric: Top N% Capture Rate
    def get_capture_rate(y_true, probas, percentile):
        # Create a DF to sort
        temp_df = pd.DataFrame({'true': y_true.values, 'proba': probas})
        temp_df = temp_df.sort_values('proba', ascending=False)
        
        # Select Top N%
        k = int(len(temp_df) * (percentile / 100))
        top_k = temp_df.head(k)
        
        # Calculate captured conversions
        captured = top_k['true'].sum()
        total = temp_df['true'].sum()
        
        return captured / total if total > 0 else 0

    capture_1 = get_capture_rate(y_test, y_proba, 1)
    capture_5 = get_capture_rate(y_test, y_proba, 5)
    capture_10 = get_capture_rate(y_test, y_proba, 10)
    capture_20 = get_capture_rate(y_test, y_proba, 20)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "train_time": train_time,
        "inference_time": inference_time,
        "Top 1% Capture": capture_1,
        "Top 5% Capture": capture_5,
        "Top 10% Capture": capture_10,
        "Top 20% Capture": capture_20
    }

def main():
    print(f"{Fore.GREEN}Loading Data...{Style.RESET_ALL}")
    df = load_data()
    
    features = ['Lead Origin', 'Lead Source', 'Total Time Spent on Website', 'Last Activity', 'Tags']
    target = 'Converted'
    
    X = df[features]
    y = df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models to Compare
    preprocessor = get_preprocessor()
    
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])
    
    lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', LGBMClassifier(random_state=42, verbose=-1))])
    
    results = []
    
    results.append(evaluate_model("Random Forest", rf_pipeline, X_train, y_train, X_test, y_test))
    results.append(evaluate_model("LightGBM", lgbm_pipeline, X_train, y_train, X_test, y_test))
    
    # Display Results
    results_df = pd.DataFrame(results)
    
    print(f"\n{Fore.YELLOW}=== Model Comparison Results ==={Style.RESET_ALL}")
    print(results_df.to_string(index=False, float_format="%.4f"))
    
    print(f"\n{Fore.GREEN}Conclusion:{Style.RESET_ALL}")
    best_acc = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"Best Accuracy: {best_acc['Model']} ({best_acc['Accuracy']:.4f})")
    
    best_speed = results_df.loc[results_df['train_time'].idxmin()]
    print(f"Fastest Training: {best_speed['Model']} ({best_speed['train_time']:.4f}s)")
    
    best_capture = results_df.loc[results_df['Top 10% Capture'].idxmax()]
    print(f"Best Top 10% Capture: {best_capture['Model']} ({best_capture['Top 10% Capture']:.4f})")

if __name__ == "__main__":
    main()
