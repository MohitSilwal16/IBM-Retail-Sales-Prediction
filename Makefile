.PHONY: dev prod

dev:
	docker compose -f docker/dev/compose.dev.yaml down -v
	docker compose -f docker/dev/compose.dev.yaml build
	docker compose -f docker/dev/compose.dev.yaml up -w

prod:
	docker compose -f docker/prod/compose.prod.yaml down
	docker compose -f docker/prod/compose.prod.yaml build
	docker compose -f docker/prod/compose.prod.yaml up -d
