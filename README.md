diff --git a/README.md b/README.md
index 246563ddba26dc1cfd1852d18e1f0ccab6266a4b..8cc06eb7826992c43a0a047202fb4ba44c75bc9a 100644
--- a/README.md
+++ b/README.md
@@ -1,8 +1,35 @@
-This is a simple proof of concept new model was created in python to model processes on Europa. The goal of this model is to assess the stability of proposed subsurface configurations while remaining a relatively simple model. 
+# Europa Ocean Model
 
-The unknown subsurface structure makes giving specific quantitative predictions as to real world configuration difficult.  
+This repository contains a simple proof-of-concept model written in Python to
+explore Europa's possible subsurface ocean. The goal is to assess the stability
+of proposed configurations while keeping the code relatively lightweight. The
+unknown structure of Europa's interior makes specific quantitative predictions
+difficult, but this model can illustrate different stability states.
 
-The model appears viable at assessing different stability states. 
+The simulation takes several planetary parameters such as geothermal heat flux,
+love number, tidal heating coefficient and hydrosphere depth. Gravity, albedo
+and solar flux are held constant in the code. Spatial and temporal resolution
+can be tuned along with the initial temperature profile.
 
-The model takes a few parameters of the planet in question, geothermal heat flux to the ocean, the love number, tidal heating coefficient, gravity, albedo, solar radiation flux and hydrosphere depth. It also uses some physical constants and takes spatial and temporal resolution specifications. Finally, an initial starting temperature profile is read into the model. 
+The project now includes a Streamlit GUI so you can watch the water column
+change over time. Sliders expose the model parameters and hover help explains
+each one.
 
+## Running the simulation
+
+Install requirements and run the model directly:
+
+```bash
+pip install -r requirements.txt
+python model.py
+```
+
+## Interactive web app
+
+To explore the model with interactive sliders run:
+
+```bash
+streamlit run app.py
+```
+
+This launches a local web page where you can tune planetary parameters and watch the temperature profile update.
