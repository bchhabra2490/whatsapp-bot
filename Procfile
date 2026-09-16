web: uvicorn voice_service:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 75 --proxy-headers --forwarded-allow-ips='*'
worker: celery -A services.tasks.celery_app worker --loglevel=info
