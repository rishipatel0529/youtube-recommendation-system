.PHONY: venv install run expand import api ui up

venv:
	python3 -m venv .venv

install: venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && python -m src.main

expand:
	. .venv/bin/activate && python -m src.expand_from_history

import:
	. .venv/bin/activate && python -m src.import_takeout /path/to/watch-history.json

api:
	. .venv/bin/activate && export PYTHONPATH=. && uvicorn api.main:app --reload --port 8000

ui:
	. .venv/bin/activate && streamlit run ui/app.py

up:
	docker-compose up --build