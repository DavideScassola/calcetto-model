venv:
		python3 -m venv .venv

.PHONY: install
install:
	python -m pip install --upgrade -r requirements.txt

.PHONY: format
format:
	black .
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place .
	isort .

.PHONY: clear_results
clear_results:
	rm -r results/*