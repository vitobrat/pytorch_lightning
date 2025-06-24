install:
	poetry install

train:
	set PYTHONPATH=. && python src/ml/train.py
