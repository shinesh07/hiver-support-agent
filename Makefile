.PHONY: setup download clean cluster golden index eval-trivial eval-simple eval-agent eval-all sweep reproduce test help

setup:
	pip install -r requirements.txt

download:
	python scripts/download.py

clean:
	python -m src.data_prep.clean_text

cluster:
	python -m src.sampling.cluster_for_stratification

golden:
	python -m src.sampling.build_golden_set

index:
	python -m src.agent.retrieval_index build

eval-trivial:
	python -m evaluate --model trivial --split evaluation

eval-simple:
	python -m evaluate --model simple --split evaluation

eval-agent:
	python -m evaluate --model agent --split evaluation

eval-all: eval-trivial eval-simple eval-agent

sweep:
	python -m src.eval.threshold_sweep

reproduce:
	bash scripts/reproduce.sh

test:
	pytest tests/ -v
