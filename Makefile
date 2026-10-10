# Feelable developer shortcuts
.PHONY: dev web build test run

dev:    ## API with auto-reload on :8000
	cd backend && python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

web:    ## Vite dev server on :5173 (proxies /api /files /samples to :8000)
	cd web && npm run dev

build:  ## Production build of the web app into web/dist
	cd web && npm run build

test:   ## Backend tests
	cd backend && python3 -m pytest -q

run:    ## Build if needed and serve everything on :8000
	./run.sh
