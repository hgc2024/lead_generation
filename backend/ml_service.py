import pandas as pd
import os
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib

# Global variable to store the model
current_model = None

# Path to data - adjusting for backend/ location
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/archive/Lead Scoring.csv'))

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def train_model(test_size=0.2, random_state=42):
    global current_model
    
    df = load_data()
    
    # Columns to use
    features = ['Lead Origin', 'Lead Source', 'Total Time Spent on Website', 'Last Activity', 'Tags']
    target = 'Converted'
    
    # Basic Cleaning
    X = df[features]
    y = df[target]
    
    # Preprocessing Pipeline
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
    
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LGBMClassifier(random_state=random_state))])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    current_model = clf
    
    # Extract Feature Importance
    # Accessing the classifier step
    model = clf.named_steps['classifier']
    importances = model.feature_importances_
    
    # Normalize importances so they sum to 1
    if importances.sum() > 0:
        importances = importances / importances.sum()
    
    # Accessing feature names from preprocessor
    onehot_columns = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(onehot_columns)
    
    # Map importance
    feature_importance_dict = dict(zip(feature_names, importances))
    
    # Sort top 10
    sorted_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)[:10])
    
    return {
        "accuracy": accuracy,
        "feature_importance": sorted_importance
    }

def predict_lead_proba(lead_data: dict):
    global current_model
    if current_model is None:
        # Auto-train if not exists
        train_model()
        
    df = pd.DataFrame([lead_data])
    proba = current_model.predict_proba(df)[0][1] # Probability of Class 1 (Converted)
    return proba

def get_leads_data():
    """Returns a sample of leads with their actual data and prediction"""
    global current_model
    if current_model is None:
        train_model()
        
    df = load_data()
    # Fill NA for display
    df = df.fillna('')
    
    # Select a subset to return to frontend
    sample = df.sample(50).to_dict(orient='records')
    
    # Add predictions
    results = []
    for record in sample:
        # We need to construct the input exactly as the model expects
        input_data = {
            'Lead Origin': record.get('Lead Origin'),
            'Lead Source': record.get('Lead Source'),
            'Total Time Spent on Website': record.get('Total Time Spent on Website') if record.get('Total Time Spent on Website') != '' else 0,
            'Last Activity': record.get('Last Activity'),
            'Tags': record.get('Tags')
        }
        
        # Calculate proba
        # Note: Input data for model needs to be a DataFrame
        proba = predict_lead_proba(input_data)
        
        # Add to record
        record['ConvertedProbability'] = float(proba)
        results.append(record)
        
    return results
