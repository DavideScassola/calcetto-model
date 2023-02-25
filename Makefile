venv:
		python3.10 -m venv venv

.PHONY: install
install:
	python -m pip install --upgrade -r requirements.txt

.PHONY: format
format:
	black .
	isort .