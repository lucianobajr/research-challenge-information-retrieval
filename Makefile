run:
	python3 -B indexer.py -m 1024 -c ./corpus/corpus_10pct.jsonl -i ./index
lint:
	pylint --rcfile=.pylintrc $(shell find . -type f -name "*.py" -not -path "*/.*" -not -path "*/venv/*")