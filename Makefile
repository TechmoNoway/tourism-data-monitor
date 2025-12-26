# Environment Setup
venv:
	python -m venv .venv
	@echo "Virtual environment created!"
	@echo "Activate it with: .venv\Scripts\Activate.ps1"

activate:
	@echo "To activate virtual environment, run:"
	@echo "  .venv\Scripts\Activate.ps1"
	@echo ""
	@echo "Or use this one-liner:"
	@echo "  .venv\Scripts\Activate.ps1; python run.py"

install:
	.venv\Scripts\python.exe -m pip install --upgrade pip
	.venv\Scripts\pip.exe install -r requirements.txt
	@echo "Dependencies installed!"

# Server
run:
	python run.py

run-venv:
	.venv\Scripts\python.exe run.py

# Collection Scripts
collect:
	python scripts/collect_data_comprehensive.py --limit 10 --analyze --report

collect-full:
	python scripts/collect_data_comprehensive.py --full-mode --all --analyze --gpu

collect-province:
	python scripts/collect_data_comprehensive.py --provinces "$(PROVINCE)" --analyze

discover:
	python scripts/collect_data_comprehensive.py --auto-discover --provinces "$(PROVINCE)" --city "$(CITY)" --dry-run

discover-save:
	python scripts/collect_data_comprehensive.py --auto-discover --provinces "$(PROVINCE)" --city "$(CITY)"

complete:
	python scripts/collect_data_comprehensive.py --complete --provinces "$(PROVINCE)"

# Analysis
analyze:
	python scripts/collect_data_comprehensive.py --analyze

analyze-gpu:
	python scripts/collect_data_comprehensive.py --analyze --gpu --batch-size 128

analyze-force:
	python scripts/collect_data_comprehensive.py --analyze --force-reanalyze --gpu

# Classification & Reports
classify:
	python scripts/collect_data_comprehensive.py --classify-types

report:
	python scripts/collect_data_comprehensive.py --report

# Combined Workflows
weekly:
	python scripts/collect_data_comprehensive.py --limit 10 --analyze --classify-types --report

monthly:
	python scripts/collect_data_comprehensive.py --full-mode --all --analyze --gpu --classify-types --report

# Utility Scripts
fix-duplicates:
	python scripts/fix_duplicates.py

fetch-images:
	python scripts/fetch_real_images.py

populate-images:
	python scripts/populate_images.py

fix-images-cloudinary:
	python scripts/fix_images_cloudinary.py

fix-images-test:
	python scripts/fix_images_cloudinary.py --test

fix-images-force:
	python scripts/fix_images_cloudinary.py --force

fix-provinces-images:
	python scripts/fix_images_cloudinary.py --provinces-only

merge-duplicates:
	python scripts/merge_duplicate_attractions.py

generate-demand:
	python scripts/generate_demand_report.py

# Database Management
clear-db:
	python scripts/clear_database.py

recreate-db:
	python scripts/recreate_db.py

restore-db:
	python scripts/restore_database.py

# Help
help:
	@echo "Tourism Data Monitor - Makefile Commands"
	@echo ""
	@echo "Environment Setup:"
	@echo "  make venv                - Create virtual environment"
	@echo "  make activate            - Show activation command"
	@echo "  make install             - Install dependencies in venv"
	@echo ""
	@echo "Server:"
	@echo "  make run                 - Start FastAPI server"
	@echo "  make run-venv            - Start server using venv Python"
	@echo ""
	@echo "Collection (Quick):"
	@echo "  make collect             - Weekly update (10 attractions, analyze, report)"
	@echo "  make collect-full        - Full collection with GPU"
	@echo "  make collect-province    - Collect for province: PROVINCE='Da Nang'"
	@echo "  make complete            - Complete workflow: PROVINCE='Da Nang'"
	@echo ""
	@echo "Discovery:"
	@echo "  make discover            - Preview discovery: PROVINCE='Quang Nam' CITY='Hoi An'"
	@echo "  make discover-save       - Save discoveries: PROVINCE='Quang Nam' CITY='Hoi An'"
	@echo ""
	@echo "Analysis:"
	@echo "  make analyze             - Analyze unanalyzed comments"
	@echo "  make analyze-gpu         - Analyze with GPU (fast)"
	@echo "  make analyze-force       - Re-analyze all comments"
	@echo ""
	@echo "Reports:"
	@echo "  make classify            - Classify tourism types"
	@echo "  make report              - Generate statistics report"
	@echo ""
	@echo "Workflows:"
	@echo "  make weekly              - Weekly update workflow"
	@echo "  make monthly             - Monthly full refresh"
	@echo ""
	@echo "Utilities:"
	@echo "  make fix-duplicates      - Fix duplicate data"
	@echo "  make fetch-images        - Fetch images for attractions"
	@echo "  make populate-images     - Populate missing images"
	@echo "  make fix-images-cloudinary - Re-upload broken images to Cloudinary"
	@echo "  make fix-images-test     - Test Cloudinary upload (5 attractions)"
	@echo "  make fix-images-force    - Re-upload ALL images to Cloudinary"
	@echo "  make fix-provinces-images - Fix province images only"
	@echo "  make merge-duplicates    - Merge duplicate attractions"
	@echo "  make generate-demand     - Generate demand report"
	@echo ""
	@echo "Database:"
	@echo "  make clear-db            - Clear database"
	@echo "  make recreate-db         - Recreate database schema"
	@echo "  make restore-db          - Restore from backup"
	@echo ""
	@echo "Examples:"
	@echo "  make collect-province PROVINCE='Da Nang'"
	@echo "  make discover PROVINCE='Quang Nam' CITY='Hoi An'"
	@echo "  make complete PROVINCE='Da Nang'"