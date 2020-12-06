PY = $(wildcard notebooks/*.py)
IPYNB := $(patsubst notebooks/%.py,notebooks/%.ipynb,$(PY))

run_notebooks: $(IPYNB)

notebooks/%.ipynb: notebooks/%.py
	@echo $@
	jupytext --to py:percent $^
	jupyter nbconvert --execute --to notebook --inplace $@
