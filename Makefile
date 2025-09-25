exclude_dirs = "experiments/"

style:
	ruff check . --fix --exclude $(exclude_dirs)
	black . --exclude $(exclude_dirs)