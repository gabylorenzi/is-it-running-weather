import sklearn
import pandas as pd 

df = pd.read_csv('activities.csv')
full_features_list = ['Weather Condition',
'Weather Temperature', #temp
'Apparent Temperature', #feels_like
'Dewpoint', 
'Humidity', #humidity
'Wind Speed', #speed
'Wind Gust', #gust
'Wind Bearing', 
'Precipitation',  
'Intensity', 
'Precipitation Probability', 
'Precipitation Type', 
'Cloud Cover',  #clouds
'Weather Visibility', 
'UV Index', 
'Weather Ozone'
]

matched_features_list = ['Weather Temperature', #temp
'Apparent Temperature', #feels_like
'Humidity', #humidity
'Wind Speed', #speed
#'Wind Gust', #gust
'Cloud Cover'  #clouds
]

target = ['Result']

all_columns = matched_features_list + target
model_df = df[all_columns].dropna()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(model_df[matched_features_list], model_df[target], test_size = 0.25, random_state = 42)
rf = RandomForestClassifier()
rf.fit(train_features, train_labels)

#OR send rf to app.py
def get_fit_model():
    return rf

predictions = rf.predict(test_features)

