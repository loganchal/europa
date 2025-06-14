# Europa Ocean Model

This repository contains a simple proof-of-concept model written in Python to
explore Europa's possible subsurface ocean. The goal is to assess the stability
of proposed configurations while keeping the code relatively lightweight. The
unknown structure of Europa's interior makes specific quantitative predictions
difficult, but this model can illustrate different stability states.

The simulation takes several planetary parameters such as geothermal heat flux,
love number, tidal heating coefficient and hydrosphere depth. Gravity, albedo
and solar flux are held constant in the code. Spatial and temporal resolution
can be tuned along with the initial temperature profile.

The project now includes a Streamlit GUI so you can watch the water column
change over time. Sliders expose the model parameters and hover help explains
each one.

## Running the simulation

Install requirements and run the model directly:

```bash
pip install -r requirements.txt
python model.py
```

## Interactive web app

To explore the model with interactive sliders run:

```bash
streamlit run app.py
```

This launches a local web page where you can tune planetary parameters and watch the temperature profile update.
