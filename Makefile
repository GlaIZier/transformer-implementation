remove-venv:
	rm -rf .venv

create-venv:
	python -m venv .venv; \
	. .venv/bin/activate; \
    python -m pip install -U pip; \

dependencies:
	. .venv/bin/activate; \
	pip install -r requirements.txt --no-cache-dir

install: remove-venv create-venv dependencies