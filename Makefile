.PHONY: install test eval ui demo
install:
	pip install -e ".[llm,ui,dev]"
test:
	pytest -q
eval:
	python eval/run_eval.py
ui:
	streamlit run ui/app.py
demo:
	diligence run "Summit Print Co" --trace
