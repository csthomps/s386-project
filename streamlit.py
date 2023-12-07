import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.title('Predicting Football Scores at Halftime')
st.subheader('Preliminary Data Visualization')

df = pd.read_csv('cbsFootballData.csv')

# selected_name = st.text_input('Enter a name','John') # John is default
# name_df = df[df['name'] == selected_name]
# if name_df.empty:
#     st.write('No data for this name')
# else:
#     fig = px.line(name_df,x='year',y='n',color='sex')
#     st.plotly_chart(fig)

fig = px.histogram(df,'final_differential',nbins = (df['final_differential'].max() - df['final_differential'].min()+1))
st.plotly_chart(fig)
    

selected_year = st.selectbox('Select a year',df['year'].unique())
selected_week = st.selectbox('Select a week',df['week'].unique())

# year_df = df[df['year'] == selected_year]
# girlnames = year_df[year_df['sex'] == 'F'].sort_values(by='n',ascending=False).head(5)['name'].reset_index(drop=True)
# boynames = year_df[year_df['sex'] == 'M'].sort_values(by='n',ascending=False).head(5)['name'].reset_index(drop=True)
# topnames = pd.concat([boynames,girlnames],axis=1)
# topnames.columns=['boy','girl']
# st.write(f'Top names in {selected_year}:')
# st.dataframe(topnames)
