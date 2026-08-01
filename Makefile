PYTHON ?= python3
PYTHONPATH := src
DEMO_DIR := results/demo

.PHONY: demo test check clean

demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tracebench demo --output $(DEMO_DIR)

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

check: test demo
	git diff --exit-code -- $(DEMO_DIR)

clean:
	find src tests -depth -type d -name __pycache__ -delete
