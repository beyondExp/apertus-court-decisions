.PHONY: run

run:
	docker build -t apertus-court-decisions .
	docker run --rm \
		-e LLM_NAME \
		-e LLM_BASE_URL \
		-e LLM_API_KEY \
		apertus-court-decisions
