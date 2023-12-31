# -*- coding: utf-8 -*-
"""Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LvponkfdufcucTjW8-Oev7y48f_CGRZV
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_path = '/content/drive/MyDrive/Project/ipl.csv'

ipl_data= pd.read_csv(data_path)

ipl_data.head()

ipl_data.describe()

ipl_data.shape

ipl_data.info()

# Check for missing values in each column
ipl_data.isnull().sum()

ipl_data.drop(['id', 'neutral_venue', 'method', 'eliminator', 'umpire1', 'umpire2'], axis=1, inplace=True)

ipl_data.isnull().sum()

# Drop rows with any missing value in the specified columns
ipl_data.dropna(subset=['city', 'player_of_match', 'winner', 'result', 'result_margin'], inplace=True)

ipl_data.info()

# Check for missing values in each column
ipl_data.isnull().sum()

ipl_data['team1'].unique()

ipl_data['team2'].unique()

ipl_data['team1'] = ipl_data['team1'].replace("Rising Pune Supergiants", "Rising Pune Supergiant")
ipl_data['team2'] = ipl_data['team2'].replace("Rising Pune Supergiants", "Rising Pune Supergiant")
ipl_data['toss_winner'] = ipl_data['toss_winner'].replace("Rising Pune Supergiants", "Rising Pune Supergiant")
ipl_data['winner'] = ipl_data['winner'].replace("Rising Pune Supergiants", "Rising Pune Supergiant")

new_data= ipl_data[['team1','team2','toss_decision','toss_winner','winner']]

new_data

top_players = ipl_data['player_of_match'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_players.index, y=top_players.values)
plt.xlabel('Player')
plt.ylabel('Number of Player of the Match Awards')
plt.title('Top Players with Most Player of the Match Awards')
plt.xticks(rotation=45)
for index, value in enumerate(top_players.values):
    plt.text(index, value, str(value), ha='center', va='bottom')
plt.show()

toss_decisions = ipl_data['toss_decision'].value_counts()
plt.pie(toss_decisions, labels=toss_decisions.index, autopct='%1.1f%%')
plt.title('Proportion of Toss Decisions')
plt.axis('equal')
plt.show()

venue_counts = ipl_data['venue'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=venue_counts.index, y=venue_counts.values)
plt.xlabel('Venue')
plt.ylabel('Number of Matches')
plt.title('Top Venues with Most Matches Played')
plt.xticks(rotation=90)
plt.show()

# Combine 'team1' and 'team2' columns to get all unique team names
teams = pd.concat([ipl_data['team1'], ipl_data['team2']])
team_wins = ipl_data['winner']

# Calculate the number of matches won by each team
team_wins_count = team_wins.value_counts()

# Plot the number of matches won by each team
plt.figure(figsize=(10, 6))
sns.barplot(x=team_wins_count.index, y=team_wins_count.values)
plt.xlabel('Team')
plt.ylabel('Number of Matches Won')
plt.title('Team Performance - Matches Won')
plt.xticks(rotation=90)
for index, value in enumerate(team_wins_count.values):
    plt.text(index, value, str(value), ha='center', va='bottom')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='toss_decision', hue='result', data=ipl_data)
plt.xlabel('Toss Decision')
plt.ylabel('Count')
plt.title('Toss Decision vs. Match Result')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(ipl_data['result_margin'], bins=30, kde=True)
plt.xlabel('Result Margin')
plt.ylabel('Frequency')
plt.title('Distribution of Result Margin')
plt.show()

#Explore the number of matches played each year to identify any trends or changes over time.
ipl_data['year'] = pd.to_datetime(ipl_data['date']).dt.year

plt.figure(figsize=(10, 6))
ax=sns.countplot(x='year', data=ipl_data, palette='viridis')
plt.xlabel('Year')
plt.ylabel('Number of Matches')
plt.title('Matches Played Each Year')
plt.xticks(rotation=45)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10)
plt.show()

city_matches = ipl_data['city'].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=city_matches.index, y=city_matches.values)
plt.xlabel('City')
plt.ylabel('Number of Matches')
plt.title('Matches Played in Each City')
plt.xticks(rotation=90)
for index, value in enumerate(city_matches.values):
    plt.text(index, value, str(value), ha='center', va='bottom')
plt.show()

"""EDA END

"""

all_teams={}
cnt=0

for index, row in ipl_data.iterrows():
    if row['team1'] not in all_teams:
        all_teams[row['team1']] = cnt
        cnt += 1
    if row['team2'] not in all_teams:
        all_teams[row['team2']] = cnt
        cnt += 1

X=new_data[['team1','team2','toss_decision','toss_winner']]
y=new_data[['winner']]

encoded_teams={w:k for k,w in all_teams.items()}

X=np.array(X)
y=np.array(y)

X

for i in range(len(X)):
  X[i][0]=all_teams[X[i][0]]
  X[i][1]=all_teams[X[i][1]]
  X[i][3]=all_teams[X[i][3]]

  y[i][0]=all_teams[y[i][0]]

X

fb={'field':0,'bat':1}

for i in range(len(X)):
  X[i][2]=fb[X[i][2]]

X

for i in range(len(X)):
  if X[i][3]==X[i][0]:
    X[i][3]=0
  else:
    X[i][3]=1

X

#if team 1 win the match then 1 if team win the match then 0
for i in range(len(y)):
  if y[i][0]==X[i][1]:
    y[i][0]=1
  else:
    y[i][0]=0

y

X=np.array(X, dtype='int')
y=np.array(y, dtype='int')

X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

test=np.array([2,4,0,1]).reshape(1,-1)
model.predict(test)

import pickle
with open('model.pkl','wb')as f:
  pickle.dump(model,f)
with open('vocab.pkl','wb')as f:
  pickle.dump(encoded_teams,f)
with open('inv_vocab.pkl','wb')as f:
  pickle.dump(all_teams,f)