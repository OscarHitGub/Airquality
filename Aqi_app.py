# FILE IMPORT #
from sklearn.ensemble import GradientBoostingRegressor
from skrub import DatetimeEncoder
import pandas as pd
import numpy as np

import streamlit as st

luchttemp = pd.read_csv("luchttemperatuur.csv")
fijnstof = pd.read_csv("fijnstof.csv")
co2 = pd.read_csv("co2.csv")
aqi = pd.read_csv("aqi.csv")
rain = pd.read_csv('CleanRain.csv')

#voeg alle data samen tot 1 dataframe
data_lucht = pd.concat(
    [luchttemp,
     fijnstof['PM 0.5'],
     co2['air_quality.mean'],
     aqi['Air Quality (ppm)']],
    axis=1)

#betere namen voor kolommen
data_lucht = data_lucht.rename(columns={
    'Air Temperature mean':'Luchttemperatuur in °C',
    'PM 0.5':'Fijnstof in ppm', 
    'air_quality.mean':'Gemiddeld CO2 gehalte in ppm', 
    'Air Quality (ppm)':'Air Quality (AQI) in ppm'
    }).dropna()

#data omzetten tot floats
data_lucht['Time'] = pd.to_datetime(data_lucht['Time'])
data_lucht['Luchttemperatuur in °C'] = data_lucht['Luchttemperatuur in °C'].str.replace(' °C', '', regex=False).astype(float)
data_lucht['Fijnstof in ppm'] = data_lucht['Fijnstof in ppm'].str.replace(' ppm', '', regex=False).astype(float)
data_lucht['Gemiddeld CO2 gehalte in ppm'] = data_lucht['Gemiddeld CO2 gehalte in ppm'].str.replace(' ppm', '', regex=False).astype(float)
data_lucht['Air Quality (AQI) in ppm'] = data_lucht['Air Quality (AQI) in ppm'].str.replace(' ppm', '', regex=False).astype(float)

#voeg de KNMI data toe
rain['Time'] = pd.to_datetime(rain['Time'])

data_lucht['Date'] = data_lucht['Time'].dt.date
rain['Date'] = rain['Time'].dt.date

data_lucht = data_lucht.merge(
    rain.drop(columns='Time'),
    on='Date',
    how='left'
)

#geen negatieve waardes
data_lucht = data_lucht[
    (data_lucht['Fijnstof in ppm'] >= 0) &
    (data_lucht['Gemiddeld CO2 gehalte in ppm'] >= 0) &
    (data_lucht['Air Quality (AQI) in ppm'] >= 0) &
    (data_lucht['Hoeveelheid regen per dag in mm'] >= 0)
]

#dag, maand, jaar ontleden uit 'Time'
dt_encoder = DatetimeEncoder()
lucht = dt_encoder.fit_transform(data_lucht["Time"])
lucht = pd.concat([lucht, 
                   data_lucht['Luchttemperatuur in °C'],
                   data_lucht['Fijnstof in ppm'],
                   data_lucht['Gemiddeld CO2 gehalte in ppm'],
                   data_lucht['Hoeveelheid regen per dag in mm']],
                  axis=1)
lucht = lucht.drop(labels='Time_total_seconds', axis=1)

#welke kolommen zorgen voor alleen maar problemen
lucht = lucht.drop(labels=['Time_hour','Time_year', 'Time_day'], axis=1)
AQI = data_lucht['Air Quality (AQI) in ppm']

#het model opstellen
model = GradientBoostingRegressor(
    n_estimators= 120,
    min_samples_split= 2,
    min_samples_leaf= 2,
    max_features= 9,
    criterion= 'squared_error',
    random_state=42)

#en fitten
model.fit(lucht, np.log1p(AQI))
feature_means = lucht.mean()

#functie om makkelijk temperatuur, maand en regenhoeveelheid in te vullen
def air_qual(temp, time_month, rain,
             fijnstof=None, co2=None):
    """
    Voorspel Air Quality (AQI) in ppm
    
    Parameters:
    temp : float
        Luchttemperatuur in °C
    time_day : int
        Dag van de maand (1–31)
    time_month : int
        Maand (1–12)
    fijnstof : float, optional
        Fijnstof in ppm (default = gemiddelde)
    co2 : float, optional
        CO2 in ppm (default = gemiddelde)
    """

    X = feature_means.copy()

    X["Luchttemperatuur in °C"] = temp
    X["Hoeveelheid regen per dag in mm"] = rain
    X["Time_month"] = time_month

    if fijnstof is not None:
        X["Fijnstof in ppm"] = fijnstof
    if co2 is not None:
        X["Gemiddeld CO2 gehalte in ppm"] = co2

    return np.expm1(model.predict(pd.DataFrame([X]))[0])

##DE APP ZELF

st.set_page_config(page_title="Luchtkwaliteit voorspellen", page_icon="🌳", layout="wide")
App_selection = st.sidebar.selectbox(
        "Extra informatie:",
        ("Luchtkwaliteit voorspellen", "Training data", "Model omschrijving")
        )

#eerste tabblad en het belangrijkste
if App_selection == "Luchtkwaliteit voorspellen":
    st.markdown(
    "<h1 style='text-align: center;'>Hoe goed is de luchtkwaliteit vandaag?</h1>",
    unsafe_allow_html=True
    )
    
    #temperatuur selection
    temperatuur = st.slider(
        "Hoeveel graden celsius is het vandaag?",
        min_value=-15,
        max_value=35,
        value=10
    )
    
    #regen selection
    regen = st.number_input("Hoeveel millimeter regen wordt er vandaag voorspeld?")
    
    #maand selection
    maanden = [
    "Januari", "Februari", "Maart", "April",
    "Mei", "Juni", "Juli", "Augustus",
    "September", "Oktober", "November", "December"
    ]
    
    maand = st.selectbox(
            "Welke maand is het?",
            ("Januari", 
             "Februari", 
             "Maart", 
             "April",
             "Mei",
             "Juni",
             "Juli",
             "Augustus",
             "September",
             "Oktober",
             "November",
             "December"
             )
    )
    maand_int = maanden.index(maand) + 1
    
    #gebruik de ingevulde data uit de app voor de voorspelling
    stream_the_AQI = air_qual(
        temp=temperatuur,
        time_month=maand_int,
        rain=regen)
    
    ppm = stream_the_AQI.round(1)
    
    #catergorie bepalen zodat het makkelijk te interpeteren is
    def bepaal_categorie(ppm):
        if ppm <= 50:
            return "Goed (0–50 ppm)", "green"
        elif ppm <= 100:
            return "Matig (51–100 ppm)", "orange"
        elif ppm <= 150:
            return "Ongezond voor gevoelige groepen (101–150 ppm)", "darkorange"
        else:
            return "Ongezond (151–200 ppm)", "red"
    
    categorie, kleur = bepaal_categorie(ppm)
    
    #laten zien nu heel duidelijk
    st.markdown(
        f"""
        <div style="height: 80vh; display: flex; flex-direction: column; 
                    justify-content: center; align-items: center;">
            <div style="font-size: 96px; color: {kleur}; margin-top: 20px; font-weight: bold;">
                {categorie}
            </div>
            <div style="font-size: 50px; color: {kleur};">
                {ppm} ppm
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )
