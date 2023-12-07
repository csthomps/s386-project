import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
import re
import time

# initialize all the columns we will use columns
game_ids = []
home_teams = []
away_teams = []
home_team_records = []
away_team_records = []
home_team_q4s = []
home_team_q3s = []
home_team_q2s = []
home_team_q1s = []
home_team_finals = []
home_team_halfs = []
away_team_q4s = []
away_team_q3s = []
away_team_q2s = []
away_team_q1s = []
away_team_finals = []
away_team_halfs = []
homewins_awaywins = []
years_list = []
weeks_list = []
home_team_nicknames = []
away_team_nicknames = []
home_first_pos = []

# define the range of years and weeks
years = range(2017, 2023) # goes from 2017 through the end of 2022
weeks = range(1,16) # goes from week 1 through the end of week 15

#! Note - this loop takes a long time to run for the full extent of the data.  For me, it took nearly 5 hours.
for year in years:
    for week in weeks:
        
        # for tracking while it runs
        print(f'Getting week {week} of {year}')
        
        # get the url for page on www.cbssports.com for the given week and season
        url = f'https://www.cbssports.com/college-football/scoreboard/FBS/{year}/regular/{week}/'
        
        # get page
        page = requests.get(url)
        soup = bs(page.content, features='lxml')
        
        # loops through each section of the page which contains data about the game
        for div in soup.find_all('div', class_=lambda x: x and x.startswith('single-score-card')):
            
            # get game_ids - formatted as NCAAF_YYYYMMDD_AWAYTEAM@HOMETEAM
            data_abbrev = div.get('data-abbrev')
            game_id = (data_abbrev)
            
            # find the table with the scores
            table = div.find('div', class_='in-progress-table section')
            #print(table)
            # get the team names
            home_team = (table.find_all('a', class_='team-name-link')[1].text.strip())
            away_team = (table.find_all('a', class_='team-name-link')[0].text.strip())
            
            # get the scores for each quarter
            game = ([td.text.strip() for td in table.find_all('td')])
            
            
            if len(game) == 12 and all(element != '-' for element in game): # check if all game data is there
                # get the quarter data and do some processing on it
                home_team_q1s.append(int(game[7]))
                home_team_q2s.append(int(game[8]))
                home_team_q3s.append(int(game[9]))
                home_team_q4s.append(int(game[10]))
                home_team_finals.append(int(game[11]))
                home_team_halfs.append(int(game[7]) + int(game[8]))
                away_team_q1s.append(int(game[1]))
                away_team_q2s.append(int(game[2]))
                away_team_q3s.append(int(game[3]))
                away_team_q4s.append(int(game[4]))
                away_team_finals.append(int(game[5]))
                away_team_halfs.append(int(game[1]) + int(game[2]))
                
                # append identifying variables to their lists
                game_ids.append(game_id)
                home_teams.append(home_team)
                away_teams.append(away_team)
                years_list.append(year)
                weeks_list.append(week)
                
                # get the team's season record, in try blocks to handle when there is no record listed
                records = table.find_all('span',class_='record')
                
                try:home_team_records.append(records[1].text.strip())
                except:home_team_records.append(pd.NA)
                
                try:away_team_records.append(records[0].text.strip())
                except:away_team_records.append(pd.NA)
                
                try:homewins_awaywins.append(int(re.findall('\d\d?',records[1].text.strip())[0]) - int(re.findall('\d\d?',records[0].text.strip())[0]))
                except: homewins_awaywins.append(pd.NA)
                
                # check if there is a play by play page
                if div.find('div', class_='bottom-bar'):
                    if 'Box Score' in div.find('div', class_='bottom-bar').text:
                        # if there is, get the url for it (using the game_id from earlier)
                        pbp_url = f'https://www.cbssports.com/college-football/gametracker/playbyplay/{game_id}'
                        
                        # get the page
                        pbp_page = requests.get(pbp_url)
                        pbp_soup = bs(pbp_page.content, features='lxml')
                        
                        # get the team nicknames (such as Cougars for BYU).  try for error handling
                        nicknames = pbp_soup.find_all('div',class_='nickname')
                        try:home_team_nicknames.append(nicknames[1].text.strip())
                        except:home_team_nicknames.append(pd.NA)
                        try:away_team_nicknames.append(nicknames[0].text.strip())
                        except:away_team_nicknames.append(pd.NA)
                        
                        # figure out which team started with the ball
                        try: 
                            first_pos = pbp_soup.find_all('span',class_='TeamName')[1].text.strip()
                            if first_pos == nicknames[1].text.strip():
                                home_first_pos.append(1)
                            else: home_first_pos.append(0)
                        except: home_first_pos.append(pd.NA)
                        
                    else: # handling if no play by play page was found
                        home_first_pos.append(pd.NA)
                        away_team_nicknames.append(pd.NA)
                        home_team_nicknames.append(pd.NA)
                else:
                    home_first_pos.append(pd.NA)
                    away_team_nicknames.append(pd.NA)
                    home_team_nicknames.append(pd.NA)


# consolidate dataframe
df = pd.DataFrame(data = {
    'year' : years_list,
    'week' : weeks_list,
    'game_id' : game_ids,
    'home_team' : home_teams,
    'home_team_nickname': home_team_nicknames,
    'away_team' : away_teams,
    'away_team_nickname': away_team_nicknames,
    'home_team_record' : home_team_records,
    'away_team_record' : away_team_records,
    'home_first_pos': home_first_pos,
    'home_team_halftime' : home_team_halfs,
    'away_team_halftime' : away_team_halfs,
    'home_team_final' : home_team_finals,
    'away_team_final' : away_team_finals,
    'home_team_q1' : home_team_q1s,
    'home_team_q2' : home_team_q2s,
    'home_team_q3' : home_team_q3s,
    'home_team_q4' : home_team_q4s,
    'away_team_q1' : away_team_q1s,
    'away_team_q2' : away_team_q2s,
    'away_team_q3' : away_team_q3s,
    'away_team_q4' : away_team_q4s,
    'homewins-awaywins': homewins_awaywins
    })

# calculate score differential at halftime
df['halftime_differential'] = df['home_team_halftime'] - df['away_team_halftime']

df['q1_differential'] = df['home_team_q1'] - df['away_team_q1']
df['q3_differential'] = df['halftime_differential'] + df['home_team_q3'] - df['away_team_q3']

# calculate score differential at end of game
df['final_differential'] = df['home_team_final'] - df['away_team_final']

#Calculating win probability for the home team based on the model outlined in this paper:http://www.glicko.net/research/nfl-chapter.pdf
from scipy.stats import norm

home_win_prob = []
expected_diff = []
for index,row in df.iterrows():
    if row['year'] > 2017:break # don't predict for the 2017 season
    home_win_prob.append(pd.NA)
    expected_diff.append(pd.NA)

for year in range(2018,2023):
    for week in range(1,16):
        print(year,week)
        
        # Create a boolean mask for rows before the target year/week
        mask = (df['year'] < year) | ((df['year'] == year) & (df['week'] < week))

        # Apply the mask to get the filtered DataFrame
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
        variance = (1/(len(filtered_df)-len(teams)-1))*np.transpose((Y-np.dot(X,thetahat)))@(Y-np.dot(X,thetahat))
        sd = np.sqrt(variance)
        
        # Create a boolean mask for rows this week
        mask = ((df['year'] == year) & (df['week'] == week))

        # Apply the mask to get the filtered DataFrame
        filtered_df2 = df[mask]
        for index,row in filtered_df2.iterrows():
            
            try: 
                mean = thetahat[teams[row['home_team']]] - thetahat[teams[row['away_team']]]
                home_win_prob.append(1-norm.cdf(0,loc=mean,scale=sd).item())
                expected_diff.append(mean.item())
                
            except:
                home_win_prob.append(pd.NA)
                expected_diff.append(pd.NA)
df['home_win_prob'] = home_win_prob
df['predicted_diff'] = expected_diff

# recalculating win probability based on only the last 15 weeks of football
home_win_prob = []
expected_diff = []
for index,row in df.iterrows():
    if row['year'] > 2017:break # don't predict for the 2017 season
    home_win_prob.append(pd.NA)
    expected_diff.append(pd.NA)

for year in range(2018,2023):
    for week in range(1,16):
        print(year,week)
        
        # Create a boolean mask for the last 15 weeks before this week
        mask = ((df['year'] == year-1) & (df['week'] >= week)) | ((df['year'] == year) & (df['week'] < week))

        # Apply the mask to get the filtered DataFrame
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
        for index,row in filtered_df.iterrows():
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
        variance = (1/(len(filtered_df)-len(teams)-1))*np.transpose((Y-np.dot(X,thetahat)))@(Y-np.dot(X,thetahat))
        sd = np.sqrt(variance)
        
        # Create a boolean mask for rows this year/week
        mask = ((df['year'] == year) & (df['week'] == week))
        # Apply the mask to get the filtered DataFrame
        filtered_df2 = df[mask]
        
        # get the number of times each team has played in the last 15 weeks
        team_counts = pd.concat([filtered_df['home_team'], filtered_df['away_team']]).value_counts()
        for index,row in filtered_df2.iterrows():
            try: 
                #make sure both teams have played at least 10 games in the last 15 (trying to avoid problems with not enough samples)
                if team_counts[row['home_team']] > 10 and team_counts[row['away_team']] > 10:
                    mean = thetahat[teams[row['home_team']]] - thetahat[teams[row['away_team']]]
                    home_win_prob.append(1-norm.cdf(0,loc=mean,scale=sd).item())
                    expected_diff.append(mean.item())
                else:
                    home_win_prob.append(pd.NA)
                    expected_diff.append(pd.NA)
            except:
                home_win_prob.append(pd.NA)
                expected_diff.append(pd.NA)
df['home_win_prob_last15'] = home_win_prob
df['predicted_diff_last15'] = expected_diff

df.to_csv('cbsFootballData.csv',index=None)