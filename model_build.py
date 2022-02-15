import sklearn
import pandas as pd 

df = pd.read_csv('activities.csv')

matched_features_list = ['Weather Temperature', #temp
'Apparent Temperature', #feels_like
'Humidity', #humidity
'Wind Speed', #speed
'Cloud Cover'  #clouds
]

target = ['Result']

all_columns = matched_features_list + target
model_df = df[all_columns].dropna()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(model_df[matched_features_list], model_df[target], test_size = 0.25, random_state = 42)

cv_grid = {'n_estimators': [50,100,200],
            'max_depth': [10,20,30],
            'min_samples_split': [2,5],
            'min_samples_leaf': [1,2,4]
        }

rf = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
cv_rf = GridSearchCV(estimator = rf, param_grid = cv_grid, cv = 3, n_jobs = 1)

cv_rf.fit(train_features, train_labels.values.ravel())

def get_fit_model():
    return cv_rf

