install:
	pip install -r requirements.txt

run:
	streamlit run app.py

predict:
	python src/predict.py

docker-build:
	docker build -t price-predictor .

docker-run:
	docker run -p 8501:8501 price-predictor
