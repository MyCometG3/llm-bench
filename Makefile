# llm-bench publication tree.
# Run make from the repository root. SRC points at the development tree that
# holds the reviewed sources; sync copies the allowlisted files into this
# tree and nothing else ever crosses over.
SRC ?= ../llm-bench

# Single source of truth for the allowlist: sync copies exactly these files,
# audit compares the staged index against exactly this set (plus repo infra).
SYNC_FILES := scripts/run_bench.py scripts/plot_summary_charts.py \
              data/prompts.json data/prompts.arith.deep.json \
              data/prompts.math.deep.json data/prompts.words.deep.json
EXPECTED_FILES := $(SYNC_FILES) Makefile README.md LICENSE .gitignore

.PHONY: publish sync stage audit

publish: sync stage audit

sync:
	@set -e; for f in $(SYNC_FILES); do mkdir -p $$(dirname $$f); cp "$(SRC)/$$f" "$$f"; done; \
	echo "synced $(words $(SYNC_FILES)) files from $(SRC)"

stage:
	git add -A

audit:
	@set -e; \
	echo "# audit 1/6: staged index equals EXPECTED_FILES (sorted set compare)"; \
	git ls-files --cached | sort > .audit.index.txt; \
	printf '%s\n' $(EXPECTED_FILES) | sort > .audit.expected.txt; \
	diff -u .audit.index.txt .audit.expected.txt || { \
		echo "audit FAILED: staged index != EXPECTED_FILES (see diff above)" >&2; \
		rm -f .audit.index.txt .audit.expected.txt; exit 1; }; \
	rm -f .audit.index.txt .audit.expected.txt; \
	echo "# audit 2/6: no unstaged differences"; \
	test -z "$$(git diff --name-only)" || { echo "audit FAILED: unstaged changes present" >&2; exit 1; }; \
	echo "# audit 3/6: no untracked files"; \
	if git status --short --untracked-files=all | grep -q '^??'; then \
		echo "audit FAILED: untracked files present" >&2; exit 1; fi; \
	echo "# audit 4/6: check-ignore effectiveness"; \
	mkdir -p data/audit_probe_tmp; \
	trap 'rm -rf data/audit_probe_tmp' EXIT; \
	git check-ignore -q data/audit_probe_tmp || { \
		echo "audit FAILED: data output dirs are not ignored" >&2; exit 1; }; \
	if git check-ignore -q data/prompts.json; then \
		echo "audit FAILED: data/prompts.json must NOT be ignored" >&2; exit 1; fi; \
	echo "# audit 5/6: forbidden-term scan (scripts/*.py data/prompts*.json README.md)"; \
	if grep -rEn 'DFlash|ornith-verify|SpecPrefill|TQKV|NgramOL|ExpEOL|\bANE\b|int8 prefill|/Volumes/|/Users/|takashi|\d{8}_\d{6}|20\d{10}' \
		scripts/*.py data/prompts*.json README.md; then \
		echo "audit FAILED: forbidden term found (see matches above)" >&2; exit 1; fi; \
	echo "# audit 6/6: dry-run smoke (no HTTP)"; \
	python3 scripts/run_bench.py --dry-run > /dev/null; \
	echo "# audit OK"
