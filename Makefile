run:
	python3 -B src/entity_search.py
v2:
	python3 -B src/v2.py
lint:
	pylint --rcfile=.pylintrc $(shell find . -type f -name "*.py" -not -path "*/.*" -not -path "*/venv/*")