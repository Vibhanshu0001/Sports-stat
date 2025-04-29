import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time

def show_model_training(df):
    st.title("Model Training")
    
    st.subheader("MVP Prediction Model")
    st.write("Train a model to predict MVP status based on player statistics")
    
    # Feature selection
    features = st.multiselect(
        "Select features for model",
        options=df.select_dtypes(include=['float64', 'int64']).columns.drop('MVP'),
        default=['Points', 'Rebounds', 'Assists', 'FG_Percentage']
    )
    
    # Model parameters
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2)
    with col2:
        n_estimators = st.slider("Number of trees", 10, 200, 100)
    
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            # Prepare data
            X = df[features]
            y = df['MVP']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            joblib.dump(model, "sports_model.pkl")
            
            time.sleep(1)  # Simulate longer processing
            
            st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")
            
            # Show classification report
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))
            
            # Feature importance
            st.subheader("Feature Importance")
            importance = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
            st.bar_chart(importance.set_index("Feature"))