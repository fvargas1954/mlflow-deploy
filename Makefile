train:
	python train.py

validate:
	python validate.py

ci: train validate

all: ci