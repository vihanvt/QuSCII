# gunicorn.config.py
timeout = 600  # increase this from default 30s to 120s
workers = 1    # can increase if needed
threads = 2    # optional, improves concurrency
