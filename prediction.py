import streamlit as st
import joblib
import pandas as pd

def show_prediction(df):
    st.title("MVP Prediction")
    
    try:
        model = joblib.load("sports_model.pkl")
        st.success("Model loaded successfully!")
    except:
        st.error("No trained model found. Please train a model first.")
        return
    
    st.subheader("Enter Player Statistics")
    
    # Get feature ranges from data
    min_points, max_points = df['Points'].min(), df['Points'].max()
    min_rebounds, max_rebounds = df['Rebounds'].min(), df['Rebounds'].max()
    min_assists, max_assists = df['Assists'].min(), df['Assists'].max()
    min_fg, max_fg = df['FG_Percentage'].min(), df['FG_Percentage'].max()
    
    col1, col2 = st.columns(2)
    
    with col1:
        points = st.slider("Points per game", min_points, max_points, min_points + 5.0)
        rebounds = st.slider("Rebounds per game", min_rebounds, max_rebounds, min_rebounds + 2.0)
    
    with col2:
        assists = st.slider("Assists per game", min_assists, max_assists, min_assists + 2.0)
        fg_percentage = st.slider("Field Goal %", min_fg, max_fg, (min_fg + max_fg)/2)
    
    if st.button("Predict MVP Probability"):
        input_data = pd.DataFrame([[points, rebounds, assists, fg_percentage]], 
                                columns=['Points', 'Rebounds', 'Assists', 'FG_Percentage'])
        
        # Get prediction and probability
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.success(f"This player has {probability:.1%} chance of being MVP!")
        else:
            st.warning(f"This player has {probability:.1%} chance of being MVP")
        
        # Show comparison with actual MVPs
        st.write("Comparison with actual MVPs:")
        mvp_stats = df[df['MVP'] == 1][['Points', 'Rebounds', 'Assists', 'FG_Percentage']].mean()
        player_stats = input_data.iloc[0]
        
        comparison = pd.DataFrame({
            'Your Player': player_stats,
            'Average MVP': mvp_stats
        })
        
        st.dataframe(comparison)