import logging
import time
from fastapi import Request

logger = logging.getLogger("research_rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

async def request_logger(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("%s %s -> %.1fms", request.method, request.url.path, elapsed)
