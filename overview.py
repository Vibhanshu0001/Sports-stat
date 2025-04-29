import streamlit as st

def show_overview(df):
    st.title("Basketball Stats Explorer")
    st.write("""
    Welcome to BasketballStatsExplorer! This application helps you analyze NBA player statistics,
    explore datasets, train models, and make predictions about player performance.
    
    ### Dataset Overview:
    This dataset contains performance metrics for top NBA players including:
    - Basic stats (Points, Rebounds, Assists)
    - Shooting percentages (FG%, 3P%, FT%)
    - Advanced metrics (Steals, Blocks, Turnovers)
    - MVP status (1=MVP, 0=Not MVP)
    """)
    
    st.subheader("Top 5 Players")
    st.dataframe(df.head())
    
    st.subheader("Key Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Players", len(df))
        st.metric("Average Points", round(df['Points'].mean(), 1))
        st.metric("Average Rebounds", round(df['Rebounds'].mean(), 1))
    
    with col2:
        st.metric("MVPs", df['MVP'].sum())
        st.metric("Average Assists", round(df['Assists'].mean(), 1))
        st.metric("Average Games Played", round(df['Games_Played'].mean(), 1))