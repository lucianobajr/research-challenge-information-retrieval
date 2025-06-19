index:
	python3 -B src/index.py
query:
	python3 -B src/query_processor.py
lint:
	pylint --rcfile=.pylintrc $(shell find . -type f -name "*.py" -not -path "*/.*" -not -path "*/venv/*")