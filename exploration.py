import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_exploration(df):
    st.title("Data Exploration")
    
    st.subheader("Dataset Preview")
    st.dataframe(df)
    
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    st.subheader("Visualizations")
    
    # Correlation heatmap
    st.write("Correlation Matrix")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Position distribution
    st.write("Position Distribution")
    fig, ax = plt.subplots()
    df['Position'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)
    
    # Team performance
    st.write("Team Performance (Average Points)")
    team_stats = df.groupby("Team")['Points'].mean().sort_values(ascending=False)
    st.bar_chart(team_stats)
    
    # MVP vs Non-MVP comparison
    st.write("MVP vs Non-MVP Comparison")
    mvp_stats = df.groupby("MVP").agg({
        'Points': 'mean',
        'Rebounds': 'mean',
        'Assists': 'mean',
        'FG_Percentage': 'mean'
    })
    st.dataframe(mvp_stats)