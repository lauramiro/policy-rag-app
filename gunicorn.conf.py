# Loading torch + the sentence-transformers model takes well over gunicorn's
# default 30s on Render's free tier, so the first /chat request after a cold
# start was killed mid-request. Raise the timeout and load the models at
# worker boot instead of inside the first request.
timeout = 120


def post_worker_init(worker):
    try:
        from app import get_resources

        get_resources()
        worker.log.info("RAG resources warmed up (vectorstore + embedding model loaded)")
    except Exception:
        # Fall back to lazy loading on the first request rather than
        # crash-looping the worker if warmup fails (e.g. missing API key).
        worker.log.exception("warmup failed; resources will load lazily")
