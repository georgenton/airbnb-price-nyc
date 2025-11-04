PY=python
export PYTHONPATH := $(PWD)


preprocess:
	$(PY) -m src.eda --input $${DATASET_PATH:-data/raw/Airbnb_Open_Data.csv} --quick-clean
	$(PY) -m src.features

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

api:
	uvicorn api.main:app --reload --port 8000

dash:
	streamlit run dashboard/streamlit_app.py
