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
nbins = int(df['final_differential'].max() - df['final_differential'].min() + 1)
st.text(f'{nbins}')
hist = px.histogram(df,'final_differential',nbins=nbins)
st.plotly_chart(hist)
    



selected_year = st.selectbox('Select a year',df['year'].unique(),2022)
selected_week = st.selectbox('Select a week',df['week'].unique(),15)

mask = (df['year'] < selected_year) | ((df['year'] == selected_year) & (df['week'] < selected_week))
filtered_df = df[mask]

# get the teams
teams = pd.concat([filtered_df['home_team'], filtered_df['away_team']]).unique()
teams_df = pd.DataFrame({'Team': sorted(teams)})
teams = {team: index for index, team in enumerate(teams_df['Team'])}

# Regression to get Theta values
n = len(filtered_df)
X = np.zeros((n,len(teams)))
Y = np.zeros((n,1))


for index, row in filtered_df.iterrows():
    home = row['home_team']
    away = row['away_team']
    # set winning team to 1 and losing team to -1
    X[index][teams[home]] = 1
    X[index][teams[away]] = -1
    Y[index] = row['final_differential']

top = np.eye(len(teams)-1,len(teams)-1)
bottom = np.zeros((1,len(teams)-1))
W = np.vstack((top,bottom))
Xstar = np.matmul(X,W)

thetahat = np.matmul(W,np.linalg.inv(np.matmul(np.transpose(Xstar),Xstar))@np.matmul(Xstar.T,Y))

thetas = pd.DataFrame({'team':list(teams.keys()),'strength':list(thetahat)})
thetas['strength'] = thetas['strength'].apply(lambda x:x.item())
thetas = thetas.sort_values('strength',ascending=False)
all_prev = px.bar(thetas.head(25),x='strength',y='team',title='Top 25 Teams Based On All Previous Data',xaxis={'categoryorder':'total descending'})
st.plotly_chart(all_prev)
