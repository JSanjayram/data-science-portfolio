.PHONY: install train test run docker-build docker-run clean

install:
	pip install -r requirements.txt

train:
	python scripts/train_model.py

test:
	python -m pytest tests/

run:
	python run.py

docker-build:
	docker build -t intelligent-assistant .

docker-run:
	docker run -p 5000:5000 intelligent-assistant

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

lint:
	flake8 src/

format:
	black src/

setup: install train

deploy: docker-build docker-compose-up