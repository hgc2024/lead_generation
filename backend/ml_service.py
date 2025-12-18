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

def engineer_features(df):
    df = df.copy()
    
    # 1. High Intent Tags (Domain Knowledge)
    high_intent_tags = ['Will revert after reading the email', 'Closed by Horizzon', 'Lost to EINS', 'Interested in other courses']
    # Handle NaN for Tags safely
    df['Tags'] = df['Tags'].fillna('Unknown')
    df['Has_High_Intent_Tag'] = df['Tags'].apply(lambda x: 1 if x in high_intent_tags else 0)
    
    # 2. Activity Score (Heuristic)
    activity_scores = {
        'SMS Sent': 10,
        'Email Opened': 8,
        'Page Visited on Website': 6,
        'Olark Chat Conversation': 5,
        'Converted to Lead': 4,
        'Email Bounced': -2,
        'Unreachable': -2
    }
    # Handle NaN for Last Activity safely
    df['Last Activity'] = df['Last Activity'].fillna('Unknown')
    df['Activity_Score'] = df['Last Activity'].map(activity_scores).fillna(0)
    
    # 3. Interaction: Time * Activity Score
    # Ensure Total Time is numeric
    df['Total Time Spent on Website'] = pd.to_numeric(df['Total Time Spent on Website'], errors='coerce').fillna(0)
    df['Time_Activity_Interaction'] = df['Total Time Spent on Website'] * df['Activity_Score']
    
    return df

def train_model(test_size=0.2, random_state=42):
    global current_model
    
    df = load_data()
    
    # Apply Feature Engineering
    df = engineer_features(df)
    
    # Updated Base Features + Engineered Features
    base_features = ['Lead Origin', 'Lead Source', 'Total Time Spent on Website', 'Last Activity', 'Tags']
    fe_features = ['Has_High_Intent_Tag', 'Activity_Score', 'Time_Activity_Interaction']
    
    features = base_features + fe_features
    target = 'Converted'
    
    # Basic Cleaning
    X = df[features]
    y = df[target]
    
    # Preprocessing Pipeline
    numeric_features = ['Total Time Spent on Website'] + fe_features
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
    
    # Apply FE for inference
    df = engineer_features(df)
    
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
