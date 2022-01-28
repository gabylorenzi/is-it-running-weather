import requests
import streamlit as st
import model_build
import numpy as np
import random

matched_features_list = ['Weather Temperature', #temp
'Apparent Temperature', #feels_like
'Humidity', #humidity
'Wind Speed', #speed
'Wind Gust', #gust
'Cloud Cover'  #clouds
]
st.title("Hey. Is it good running weather?")
st.write("The model uses real temperature, feels-like temperature, humidity, wind speed, and cloud cover to analyze running conditions.")
st.write("")

city_name = st.text_input("City Name")
url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid=29dabdd9237ba71d01601da27f3d052b&units=metric"
response = requests.get(url)

class weather():
  def __init__(self, city_name, response):
    self.city = city_name
    self.temp = response.json()['main']['temp']
    self.feels_like = response.json()['main']['feels_like']
    self.humidity = response.json()['main']['humidity']*.01
    self.wind_speed = response.json()['wind']['speed']
    self.cloud_cover = response.json()['clouds']['all']*.01

if(city_name):
  pred_weather = weather(city_name,response)

  predict_features = [pred_weather.temp, pred_weather.feels_like, pred_weather.humidity, pred_weather.wind_speed,pred_weather.cloud_cover]

  predict_features = np.array(predict_features).reshape(1,-1)
  rf = model_build.get_fit_model()
  result = str(rf.predict(predict_features)[0])
  res_print = f"It is **{result}** running weather."
  st.write(res_print)

  running_quotes = [
    "“To give anything less than your best, is to sacrifice the gift.” - **Steve Prefontaine**",
    "“I don’t run to add days to my life, I run to add life to my days.” — **Ronald Rook**",
    "“We are what we repeatedly do. Excellence, then, is not an act, but a habit.” — **Aristotle**",
    "“Run often. Run long. But never outrun your joy of running.” — **Julie Isphording**",
  ]
  st.write("")
  st.write("")
  st.write(random.choice(running_quotes))
