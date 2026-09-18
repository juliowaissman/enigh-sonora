# Tablero ENIGH · Ingreso de las familias en Sonora
#
# El intérprete de Python vive dentro del proyecto (.tools/) porque el entorno no
# permite escribir en /opt/homebrew ni en ~/.local. `make venv` lo detecta y crea
# el entorno virtual con él.
#
# Uso rápido:
#   make install     # crea el venv e instala dependencias
#   make data        # descarga y procesa la ENIGH (2020, 2022, 2024)
#   make app         # levanta el tablero

PYTHON ?= $(shell ls .tools/python3.12/Versions/3.12/bin/python3.12 2>/dev/null || which python3.12 || which python3)
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: help install venv data data-fast geo app test lint fmt validate clean clean-data

help:
	@echo "Objetivos disponibles:"
	@echo "  install      Crea el entorno virtual e instala dependencias"
	@echo "  data         Descarga la ENIGH y calcula todos los agregados (lento)"
	@echo "  data-fast    Igual, pero sin intervalos de confianza (rápido)"
	@echo "  geo          Solo descarga la cartografía de los mapas"
	@echo "  app          Levanta el tablero Streamlit"
	@echo "  test         Ejecuta las pruebas"
	@echo "  lint         Revisa el estilo con ruff"
	@echo "  validate     Valida los datos procesados contra cifras de control"
	@echo "  clean        Borra cachés"
	@echo "  clean-data   Borra los datos crudos y procesados"

install: venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

venv:
	@if [ ! -x "$(PY)" ]; then \
		echo "Creando entorno virtual con $(PYTHON)"; \
		$(PYTHON) -m venv $(VENV); \
	fi
	@$(PY) -V

data: install
	$(PY) -m enigh.pipeline

data-fast: install
	$(PY) -m enigh.pipeline --sin-bootstrap

geo: install
	$(PY) -m enigh.pipeline --solo-geografia

app: install
	$(VENV)/bin/streamlit run app.py

test: install
	$(PY) -m pytest tests/ -v

lint: install
	$(PY) -m ruff check enigh/ tests/ app.py

fmt: install
	$(PY) -m ruff check --fix enigh/ tests/ app.py

validate: install
	$(PY) -m enigh.validacion

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

clean-data:
	rm -rf data/raw data/interim data/processed
