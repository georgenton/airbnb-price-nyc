PY=python

preprocess:
	$(PY) src/eda.py --input $${DATASET_PATH:-data/raw/Airbnb_Open_Data.csv} --quick-clean
	$(PY) src/features.py

train:
	$(PY) src/train.py

evaluate:
	$(PY) src/evaluate.py

api:
	uvicorn api.main:app --reload --port 8000

dash:
	streamlit run dashboard/streamlit_app.py
