.PHONY: help install run-api run-dashboard test lint format \
        docker-build docker-up docker-down docker-logs \
        migrate migrate-new retrain

help:
	@echo "Content Moderation Pipeline — common commands"
	@echo "  make install          Install Python dependencies"
	@echo "  make run-api          Run the FastAPI backend locally (uvicorn --reload)"
	@echo "  make run-dashboard    Run the Streamlit dashboard locally"
	@echo "  make test             Run the pytest suite with coverage"
	@echo "  make lint             Run ruff + mypy"
	@echo "  make format           Run black + ruff --fix"
	@echo "  make docker-build     Build all Docker images"
	@echo "  make docker-up        Start the full stack (postgres, mlflow, api, dashboard, retrain scheduler)"
	@echo "  make docker-down      Stop the stack"
	@echo "  make docker-logs      Tail logs from all services"
	@echo "  make migrate          Apply Alembic migrations (upgrade head)"
	@echo "  make migrate-new msg=\"...\"  Autogenerate a new Alembic migration"
	@echo "  make retrain          Run the retraining pipeline once, locally"

install:
	pip install -r requirements.txt

run-api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py

test:
	pytest --cov=app --cov-report=term-missing

lint:
	ruff check .
	mypy app

format:
	black .
	ruff check --fix .

docker-build:
	docker compose build

docker-up:
	docker compose up -d
	@echo "API:       http://localhost:8000/docs"
	@echo "Frontend:  http://localhost:3000"
	@echo "Dashboard: http://localhost:8501"
	@echo "MLflow:    http://localhost:5000"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

migrate:
	alembic upgrade head

migrate-new:
	alembic revision --autogenerate -m "$(msg)"

retrain:
	python retrain.py
