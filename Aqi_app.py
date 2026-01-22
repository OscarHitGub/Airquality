# FILE IMPORT #
from sklearn.ensemble import GradientBoostingRegressor
from skrub import DatetimeEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

luchttemp = pd.read_csv("luchttemperatuur.csv")
aqi = pd.read_csv("aqi.csv")
rain = pd.read_csv('CleanRain.csv')
wind = pd.read_csv('CleanWind.csv')

#voeg alle data samen tot 1 dataframe
data_lucht = pd.concat(
    [luchttemp,
     aqi['Air Quality (ppm)']],
    axis=1)

#betere namen voor kolommen
data_lucht = data_lucht.rename(columns={
    'Air Temperature mean':'Luchttemperatuur in °C',
    'Air Quality (ppm)':'Air Quality (AQI) in ppm'
    }).dropna()

#data omzetten tot floats
data_lucht['Time'] = pd.to_datetime(data_lucht['Time'])
data_lucht['Luchttemperatuur in °C'] = data_lucht['Luchttemperatuur in °C'].str.replace(' °C', '', regex=False).astype(float)
data_lucht['Air Quality (AQI) in ppm'] = data_lucht['Air Quality (AQI) in ppm'].str.replace(' ppm', '', regex=False).astype(float)

#dag of nacht
data_lucht["Time"] = pd.to_datetime(data_lucht["Time"], errors="coerce")

data_lucht["Hour"] = data_lucht["Time"].dt.hour
data_lucht["dag_nacht"] = np.where((data_lucht["Hour"] >= 7) & (data_lucht["Hour"] < 19), "Dag", "Nacht")

#voeg de KNMI data toe
rain['Time'] = pd.to_datetime(rain['Time'])

data_lucht['Date'] = data_lucht['Time'].dt.date
rain['Date'] = rain['Time'].dt.date

data_lucht = data_lucht.merge(
    rain.drop(columns='Time'),
    on='Date',
    how='left'
)

wind['Time'] = pd.to_datetime(wind['Time'])
wind['Date'] = wind['Time'].dt.date

data_lucht = data_lucht.merge(
    wind.drop(columns='Time'),
    on='Date',
    how='left'
)

#geen negatieve waardes
data_lucht = data_lucht[
    (data_lucht['Air Quality (AQI) in ppm'] >= 0) &
    (data_lucht['Windkracht'] >= 0) &
    (data_lucht['Hoeveelheid regen per dag in mm'] >= 0)
]

#dag, maand, jaar ontleden uit 'Time'
dt_encoder = DatetimeEncoder()
lucht = dt_encoder.fit_transform(data_lucht["Time"])
lucht = pd.concat([lucht, 
                   data_lucht['Luchttemperatuur in °C'],
                   data_lucht['Windkracht'],
                   data_lucht['Hoeveelheid regen per dag in mm']],
                  axis=1)
lucht = lucht.drop(labels='Time_total_seconds', axis=1)

#welke kolommen zorgen voor alleen maar problemen
lucht = lucht.drop(labels=['Time_hour','Time_year', 'Time_day'], axis=1)
AQI = data_lucht['Air Quality (AQI) in ppm']

#het model opstellen
model = GradientBoostingRegressor(
    n_estimators= 230,
    min_samples_split= 7,
    min_samples_leaf= 1,
    max_features= 6,
    criterion= 'squared_error',
    random_state=42)

#en fitten
model.fit(lucht, np.log1p(AQI))
feature_means = lucht.mean()

#functie om makkelijk temperatuur, maand en regenhoeveelheid in te vullen
def air_qual(temp, time_month, rain, airspeed):
    """
    Voorspel Air Quality (AQI) in ppm
    
    Parameters:
    temp : float
        Luchttemperatuur in °C
    time_day : int
        Dag van de maand (1–31)
    time_month : int
        Maand (1–12)
    """

    X = feature_means.copy()

    X["Luchttemperatuur in °C"] = temp
    X["Hoeveelheid regen per dag in mm"] = rain
    X["Time_month"] = time_month
    X["Windkracht"] = airspeed

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
    
    #wind selection
    windsnelheid = st.slider(
        "Wat is de voorspelde windkracht vandaag?",
        min_value=0,
        max_value=12,
        value=0
    )
    
    #gebruik de ingevulde data uit de app voor de voorspelling
    stream_the_AQI = air_qual(
        temp=temperatuur,
        time_month=maand_int,
        rain=regen,
        airspeed=windsnelheid)
    
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

if App_selection == "Training data":
    st.set_page_config(layout="centered")
    scatterselect = st.radio(
        "Selecteer data om te visualizeren:", 
        ("Temperatuur", "Regen", "Windkracht")
    )
    if scatterselect == "Temperatuur":
        fig, ax = plt.subplots()
        
        ax.scatter(
            data_lucht["Luchttemperatuur in °C"], 
            data_lucht["Air Quality (AQI) in ppm"], 
            alpha=0.4,
            color="maroon"
        )
        
        ax.set_title("Temperatuur tegen luchtkwaliteit")
        ax.set_xlabel("Temperatuur (°C)")
        ax.set_ylabel("Luchtkwaliteit (ppm)")
        
        st.pyplot(fig)
        
        st.text(
            "Er is geen duidelijk verband zichtbaar tussen de temperatuur en de luchtkwaliteit.",
            text_alignment="center",
            width="stretch"
        )

    if scatterselect == "Regen":
        fig, ax = plt.subplots()
        
        ax.scatter(
            data_lucht["Hoeveelheid regen per dag in mm"], 
            data_lucht["Air Quality (AQI) in ppm"], 
            alpha=0.4,
            color="darkcyan"
        )
        
        ax.set_title("Hoeveelheid regen per dag tegen luchtkwaliteit")
        ax.set_xlabel("Hoeveelheid regen per dag (mm)")
        ax.set_ylabel("Luchtkwaliteit (ppm)")
        
        st.pyplot(fig)
        
        st.text(
            "Er is geen duidelijk verband zichtbaar tussen hoeveelheid regen per dag en de luchtkwaliteit.",
            text_alignment="center",
            width="stretch"
        )
    
    if scatterselect == "Windkracht":
        fig, ax = plt.subplots()
        
        ax.scatter(
            data_lucht["Windkracht"], 
            data_lucht["Air Quality (AQI) in ppm"], 
            alpha=0.4,
            color="navy"
        )
        ax.plot(
            data_lucht.groupby("Windkracht")["Air Quality (AQI) in ppm"].median(),
            color="slategrey",
            label="Mediaan"
        )
        ax.plot(
            data_lucht.groupby("Windkracht")["Air Quality (AQI) in ppm"].mean(),
            color="slateblue",
            label="Gemiddelde"
        )
        
        ax.set_title("Wind tegen luchtkwaliteit")
        ax.set_xlabel("Windkracht")
        ax.set_ylabel("Luchtkwaliteit (ppm)")
        ax.legend()
        
        st.pyplot(fig)
        
        st.text(
            "Bij een hogere windkracht is de luchtkwaliteit gemiddeld beter.",
            text_alignment="center",
            width="stretch"
        )

    st.divider()
    
    fig, ax = plt.subplots()
    
    data_lucht.boxplot(
        column="Air Quality (AQI) in ppm",
        by="dag_nacht",
        ax=ax
    )
    
    ax.set_title("Luchtkwaliteit overdag tegen ’s nachts")
    ax.set_xlabel("")
    ax.set_ylabel("Luchtkwaliteit (ppm)")
    plt.suptitle("")
    
    st.pyplot(fig)
    
    st.text(
        "De verdeling van luchtkwaliteit verschilt tussen de dag en de nacht.",
        text_alignment="center",
        width="stretch"
    )
    st.text("Bron voor temperatuur: https://www.mt-dashboard.nl/admin/d/xbR4dVq4z")
    st.text("Bron voor regen en wind: https://www.knmi.nl/nederland-nu/klimatologie/daggegevens")
if App_selection == "Model omschrijving":
    st.markdown("""
## Modelbeschrijving – Gradient Boosting Regressor

Het uiteindelijke model dat gebruikt is voor deze voorspelling, is geschreven in Python en maakt gebruik van de machine learning methode **Gradient Boosting**.  
Deze methode werkt met *decision trees* en de fouten (*errors*) van eerdere trees om steeds betere voorspellingen te maken.

Het proces begint met het trainen van één decision tree.  
Daarna wordt een volgende tree getraind om de fouten van de vorige tree te compenseren.  
Dit proces herhaalt zich zo vaak als is ingesteld met het aantal estimators.

Het model maakt gebruik van de Python package **Scikit-learn**, en daarvan de functie `GradientBoostingRegressor`.

### Gebruikte hyperparameters
De volgende hyperparameters zijn gebruikt:
- `n_estimators`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `criterion`

### Bepalen van de hyperparameters
Om de optimale waarden voor deze hyperparameters te bepalen, is een **randomized gridsearch** uitgevoerd.

Voor alle hyperparameters behalve `criterion` is een range van integerwaarden opgegeven (bijvoorbeeld 0 t/m 12).  
Voor `criterion` is gekozen tussen:
- `friedman_mse`
- `squared_error`

De randomized gridsearch selecteert willekeurige combinaties binnen deze ranges en test per combinatie de fout.  
Uiteindelijk worden de hyperparameters gekozen die de laagste fout opleveren.  
Deze combinatie is optimaal **als geheel**, specifiek voor dit model.

### Optimale hyperparameterwaarden
- `n_estimators`: **230**  
- `min_samples_split`: **7**  
- `min_samples_leaf`: **1**  
- `max_features`: **6**  
- `criterion`: **friedman_mse**

### Uitleg hyperparameters
`n_estimators` bepaalt hoe vaak de Gradient Boosting-cyclus wordt herhaald.

Een splitsing binnen een decision tree wordt bepaald door `min_samples_split`, wat aangeeft hoeveel samples minimaal nodig zijn om een splitsing te maken.

Wanneer verdere splitsingen niet nodig zijn, ontstaat een *leaf node*.  
`min_samples_leaf` bepaalt hoeveel samples minimaal nodig zijn om zo’n leaf node te vormen.

`max_features` bepaalt hoeveel features mogen worden meegenomen bij het bepalen van een splitsing.

Het type foutmaat wordt bepaald door `criterion`:
- `squared_error` komt overeen met de **Mean Squared Error (MSE)**
- `friedman_mse` is een variant hierop met een *Friedman improvement score*

---

## Trainen van het model

Na het instellen van het model met bovenstaande hyperparameters is het getraind op data van:
- temperatuur  
- regen  
- windkracht  

op specifieke tijdstippen.

Bij het trainen zijn de volgende evaluatiematen berekend:
- **MAE** (Mean Absolute Error)  
- **RMSE** (Root Mean Squared Error)  
- **R²-score**

De MAE is vergelijkbaar met de MSE, maar gebruikt de absolute fout in plaats van het kwadraat.  
De RMSE is de wortel van de MSE.  
Voor zowel MAE als RMSE geldt: **lager is beter**.

De R²-score geeft aan hoe accuraat het model is.  
Een R² van 0.5 betekent dat het model 50% van de variatie verklaart.  
Deze waarde is idealiter zo dicht mogelijk bij 1.

### Resultaten
Na het trainen en evalueren van het model zijn de resultaten:
- **RMSE**: 35.12  
- **MAE**: 27.66  
- **R²**: 0.28  

Dit betekent dat het model ongeveer **30% van de waarden correct voorspelt**.

Als het model bijvoorbeeld **75 ppm** voorspelt, kan de werkelijke luchtkwaliteit tussen **47 ppm en 102 ppm** liggen.  
Dit verschil is groot en betekent dat een voorspelling van *goede luchtkwaliteit* (0–50 ppm) in werkelijkheid ook *matige luchtkwaliteit* (51–100 ppm) kan zijn.

Bron:  
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
""")
    
    st.markdown("""
## RandomForest Regressor

Een van de modellen die onderzocht maar niet gebruikt is, is een **RandomForest model**.  
Dit model werkt door het maken van meerdere *decision trees*, waarbij iedere tree een kleine afwijking is van de vorige.  
Na het maken van alle decision trees wordt er een gemiddelde genomen van alle trees en dit gemiddelde wordt gebruikt als voorspelling.

Het model gebruikt de Python package **Scikit-learn**, en daarvan de functie `RandomForestRegressor`.

### Gebruikte hyperparameters
De volgende hyperparameters zijn gebruikt:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `bootstrap`

Een aantal van deze hyperparameters wordt ook gebruikt door het **Gradient Boosting** model, maar met andere optimale waarden:
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

`n_estimators` wordt ook door beide modellen gebruikt, maar werkt net anders bij beide modellen.

De optimale waarden zijn bepaald met een **GridSearch**, die verschillende combinaties test om de beste configuratie te vinden.

### Optimale hyperparameterwaarden
- `min_samples_split`: **4**
- `min_samples_leaf`: **1**
- `max_features`: **log2**
- `n_estimators`: **450**
- `max_depth`: **30**
- `bootstrap`: **True**

`bootstrap` zorgt ervoor dat iedere tree een iets andere dataset ziet, waardoor meer variatie ontstaat.  
Dit voorkomt dat fouten van één tree worden doorgegeven aan andere trees en verbetert de robuustheid van het model.

### Trainen van het model
Het RandomForest-model is getraind op data van:
- temperatuur  
- regen  
- windkracht  

op verschillende tijdstippen.  
Hierbij zijn de **MAE**, **RMSE** en **R²-score** berekend.

### Resultaten
Na het trainen en evalueren van het model zijn de resultaten:
- **RMSE**: 36.31  
- **MAE**: 27.74  
- **R²**: 0.27  

Dit betekent dat het model ongeveer **27% van de variatie** in de data verklaart.  
Als het model bijvoorbeeld **50 ppm** voorspelt, kan de werkelijke waarde tussen **22 en 77 ppm** liggen.  
Hierdoor kan een voorspelling van *matige luchtkwaliteit* (51–100 ppm) in werkelijkheid *goede luchtkwaliteit* (0–50 ppm) zijn.

Bron:  
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html


---

## K-Nearest Neighbors Regressor

Een ander onderzocht maar niet gebruikt model is het **K-Nearest Neighbors (KNN)** model.  
Dit model werkt door de dichtstbijzijnde trainingspunten te zoeken en hiervan een gemiddelde te nemen om een voorspelling te maken.

Het model gebruikt de functie `KNeighborsRegressor` uit de **Scikit-learn** package.

### Gebruikte hyperparameters
- `n_neighbors`
- `weights`
- `metric`
- `p`

### Optimale hyperparameterwaarden
- `n_neighbors`: **25**
- `weights`: **distance**
- `metric`: **manhattan**
- `p`: **1**

Bij `weights='distance'` krijgen dichterbij liggende punten meer invloed.  
De **Manhattan distance** telt de absolute verschillen van alle variabelen bij elkaar op.  
Een `p`-waarde van 1 werkt optimaal in combinatie met deze metric.

### Trainen van het model
Net als bij de andere modellen is KNN getraind op:
- temperatuur  
- regen  
- windkracht  

Hierbij zijn opnieuw de **MAE**, **RMSE** en **R²-score** berekend.

### Resultaten
Na evaluatie zijn de prestaties:
- **RMSE**: 36.94  
- **MAE**: 29.38  
- **R²**: 0.20  

Dit betekent dat het model slechts **20% van de variatie** correct voorspelt.  
Bij een voorspelling van **50 ppm** kan de werkelijke waarde tussen **20 en 80 ppm** liggen.  
Ook hier kan een voorspelling van *matige luchtkwaliteit* in werkelijkheid *goede luchtkwaliteit* zijn.

Bron:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
""")













