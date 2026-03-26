# Medical Insurance Cost Prediction App

This is a Streamlit web app that predicts medical insurance charges based on:

- Age
- Sex
- BMI
- Children
- Smoker
- Region

## Files in this project

- `train_model.py` → trains the machine learning model
- `app.py` → Streamlit web app
- `requirements.txt` → required Python libraries
- `insurance.csv` → dataset
- `best_insurance_model.joblib` → trained model file

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
