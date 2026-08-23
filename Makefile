.PHONY: test test-all help

# dm0049-client is a test dependency only: the point of the suite is that a
# file from any generation loads through the reader a consumer actually uses.
PYTEST_FLAGS := -x -q -p randomly --tb=short

test:
	python -m pytest $(PYTEST_FLAGS) test_nc_upgrade.py

test-all: test

help:
	@echo ""
	@echo "Targets:"
	@echo "  make test      - Run the test suite (random order, first failure stops)"
	@echo "  make test-all  - Every test target"
	@echo ""
	@echo "Adding a container generation: acquire a file with the current"
	@echo "client, drop it in resource/nc/, add its name and sha1 to FIXTURES"
	@echo "in test_nc_upgrade.py. Nothing else changes."
