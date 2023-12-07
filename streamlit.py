import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.title('Popular Names')

url = 'https://github.com/esnt/Data/raw/main/Names/popular_names.csv'
df = pd.read_csv(url)

selected_name = st.text_input('Enter a name','John') # John is default
name_df = df[df['name'] == selected_name]
if name_df.empty:
    st.write('No data for this name')
else:
    fig = px.line(name_df,x='year',y='n',color='sex')
    st.plotly_chart(fig)
    

selected_year = st.select_slider('Select a year',df['year'].unique())
year_df = df[df['year'] == selected_year]
girlnames = year_df[year_df['sex'] == 'F'].sort_values(by='n',ascending=False).head(5)['name'].reset_index(drop=True)
boynames = year_df[year_df['sex'] == 'M'].sort_values(by='n',ascending=False).head(5)['name'].reset_index(drop=True)
topnames = pd.concat([boynames,girlnames],axis=1)
topnames.columns=['boy','girl']
st.write(f'Top names in {selected_year}:')
st.dataframe(topnames)
