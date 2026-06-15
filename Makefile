.PHONY: install test run run-cli

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

run:
	streamlit run app.py

run-cli:
	python main.py
