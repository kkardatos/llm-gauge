# Benchmark local LLM models

## Install dependency
```bash
pip install -r requirements
``` 

## Edit prompts.txt
```text
Why is the sky blue?
Explain quantum computing.
Write a Python function to sort a list.
What are the advantages of Rust?
``` 

## Run
```bash
python llm_benchmark.py --url http://localhost:11434/api/generate --models qwen3-coder:latest qwen3.5:9b --prompts-file prompts.txt --output report.html
```