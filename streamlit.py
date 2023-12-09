import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.title('Predicting Football Scores at Halftime')
st.subheader('Preliminary Data Visualization')

df = pd.read_csv('cbsFootballData.csv')

st.write("This is an interactive dashboard to explore the charts I wrote about in this blog post: https://csthomps.github.io/s386-blog/2023/12/06/386-project-part-2.html")
st.write("I will breifly describe the purpose and data behind each of the visualizations, but look at the blog post for more detail.")

st.write("This chart shows the distribution of the final differentials across all the games in my dataset.  Differentials are calculated as the home team's score minus the away team's score.  As you might expect, there are large spikes at the threes and sevens.  However, there definitely seems to be a majority of the observations on the positive side, which indicates that the home team wins more often.")
nbins = int(df['final_differential'].max() - df['final_differential'].min() + 1)
hist = px.histogram(df,'final_differential',nbins=nbins)
st.plotly_chart(hist)
    

st.write("This chart shows the relationship between two of the most important variables, halftime differential and final differential.  Since my ultimate goal is to predict the final differential based on the halftime differential, the fact that this relationship is so linear is a good sign.")
scatter = px.scatter(df,x='halftime_differential',y='final_differential',
           opacity=.2,hover_data=["home_team",
                                  "away_team",
                                  'halftime_differential',
                                  'final_differential'])
st.plotly_chart(scatter,use_container_width=True)

st.write('These next two plots show the relationships between all the variables in the dataset.  I have set it up to allow you to interactively choose which variables to look at. ')
st.write("First, here is a quick explanation of what the variables are:")
st.write("- homewins-awaywins: the number of games the home team has won so far in the season - the away team's win count")
st.write("- home_win_prob and predicted_diff: these are calculated using through Glickman and Stern's method (which I talk about in my blog posts) and all historical data")
st.write("- home_win_prob_last15 and predicted_diff_last15: these are calculated using only the last 15 weeks of data")
st.write("- q1_differential, halftime_differential, q3_differential, final_differential: the score differential (home score - away score) at each point in the game")

variables = st.multiselect(
    'What variables do you want to include?',
    ['homewins-awaywins','home_win_prob','predicted_diff','home_win_prob_last15',
                      'predicted_diff_last15','q1_differential','halftime_differential',
                      'q3_differential','final_differential'],
    ['home_win_prob_last15','predicted_diff_last15','halftime_differential','final_differential'])

heatmap = px.imshow(df[variables].corr().round(2),
          text_auto=True,
          title='Correlation Heatmap')
st.plotly_chart(heatmap,use_container_width=True)

num_vars = len(variables)
scat_mat = px.scatter_matrix(df[variables],
                             height = 200*num_vars, width = 200*num_vars,
                             title = 'Scatterplot Matrix',
                             opacity=.2)
st.plotly_chart(scat_mat)


st.write("In my opinion, one of the most interesting variables I have gathered in this dataset are the pre-game predictions based on past team performance.  I won't go into the technical details here, but essentially the method estimates each team's relative strength, then uses that to predict who will win and by how many points.")
st.write("I have found that, for some teams, team strength varies greatly over seasons while for others it stays relatively constant.  That can be seen clearly by comparing these two plots.  I set it up to allow you to look at the strongest 25 teams at any given time point in the data, based solely on the previous games.")

selected_year = st.selectbox('Select a year',df['year'].unique(),len(df['year'].unique())-1)
selected_week = st.selectbox('Select a week',df['week'].unique(),len(df['week'].unique())-1)

if selected_year == 2017 and selected_week == 1:
    st.write("There is no data before that!")
elif selected_year == 2017 and selected_week < 5:
    st.write("There isn't enough data before that to do regression.")
else:
    ## All Previous
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
    all_prev = px.bar(thetas.head(25),x='strength',y='team',height=600)
    all_prev.update_layout(yaxis=dict(autorange="reversed"))



    ## Last 15
    mask = ((df['year'] == selected_year-1) & (df['week'] >= selected_week)) | ((df['year'] == selected_year) & (df['week'] < selected_week))
    filtered_df = df[mask]

    # get the teams
    teams = pd.concat([filtered_df['home_team'], filtered_df['away_team']]).unique()
    teams_df = pd.DataFrame({'Team': sorted(teams)})
    teams = {team: index for index, team in enumerate(teams_df['Team'])}

    # Regression to get Theta values
    n = len(filtered_df)
    X = np.zeros((n,len(teams)))
    Y = np.zeros((n,1))


    i = 0
    for index, row in filtered_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        # set winning team to 1 and losing team to -1
        X[i][teams[home]] = 1
        X[i][teams[away]] = -1
        Y[i] = row['final_differential']
        i += 1

    top = np.eye(len(teams)-1,len(teams)-1)
    bottom = np.zeros((1,len(teams)-1))
    W = np.vstack((top,bottom))
    Xstar = np.matmul(X,W)

    thetahat = np.matmul(W,np.linalg.inv(np.matmul(np.transpose(Xstar),Xstar))@np.matmul(Xstar.T,Y))

    thetas = pd.DataFrame({'team':list(teams.keys()),'strength':list(thetahat)})
    thetas['strength'] = thetas['strength'].apply(lambda x:x.item())
    thetas = thetas.sort_values('strength',ascending=False)
    last_15 = px.bar(thetas.head(25),x='strength',y='team',height=600)
    last_15.update_layout(yaxis=dict(autorange="reversed"))

    col1, col2 = st.columns(2)

    col1.header("Top 25 Teams Based On All Previous Data")
    col1.plotly_chart(all_prev,use_container_width=True)

    col2.header("Top 25 Teams Based On last 15 weeks")
    col2.plotly_chart(last_15,use_container_width=True)

st.write("It is also interesting to compare the results of these models with other ranking systems, such as the AP poll.")

df_2022 = df[df['year'] == 2022]
# get the teams
teams_2022 = pd.concat([df_2022['home_team'], df_2022['away_team']]).unique()
teams_df_2022 = pd.DataFrame({'Team': sorted(teams_2022)})
teams_2022 = {team: index for index, team in enumerate(teams_df_2022['Team'])}

# Regression to get Theta values
n_2022 = len(df_2022)
X_2022 = np.zeros((n_2022,len(teams_2022)))
Y_2022 = np.zeros((n_2022,1))


i = 0
for index, row in df_2022.iterrows():
    home = row['home_team']
    away = row['away_team']
    # set winning team to 1 and losing team to -1
    X_2022[i][teams_2022[home]] = 1
    X_2022[i][teams_2022[away]] = -1
    Y_2022[i] = row['final_differential']
    i += 1

top_2022 = np.eye(len(teams_2022)-1,len(teams_2022)-1)
bottom_2022 = np.zeros((1,len(teams_2022)-1))
W_2022 = np.vstack((top_2022,bottom_2022))
Xstar_2022 = np.matmul(X_2022,W_2022)

thetahat_2022 = np.matmul(W_2022,np.linalg.inv(np.matmul(np.transpose(Xstar_2022),Xstar_2022))@np.matmul(Xstar_2022.T,Y_2022))

thetas_2022 = pd.DataFrame({'team':list(teams_2022.keys()),'strength':list(thetahat_2022)})
thetas_2022['strength'] = thetas_2022['strength'].apply(lambda x:x.item())
thetas_2022 = thetas_2022.sort_values('strength',ascending=False)
season_2022 = px.bar(thetas_2022.head(25),x='strength',y='team',height=600)
season_2022.update_layout(yaxis=dict(autorange="reversed"))

col1, col2 = st.columns(2)

col1.header("Estimated Top 25 Teams in 2022")
col1.plotly_chart(season_2022,use_container_width=True)

col2.header("AP Top 25 Teams in 2022")
col2.image('s386-project\\figures\\Screenshot 2023-12-08 231850.png')
