.PHONY: install update format lint test sec exp-req

install:
	@poetry install
update:
	@poetry update	
format:
	@poetry run black .
	@poetry run isort .
	@poetry run pydocstyle .
	@poetry run prospector --no-autodetect
lint:
	@poetry run darker --check .
	@poetry run darker --isort .
sec:
	@poetry run pip-audit
exp-req:
	@poetry export -f requirements.txt --without-hashes  > requirements.txt
	@poetry export --dev -f  requirements.txt --without-hashes  > dev-requirements.txt
