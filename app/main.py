from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from app.core import config
from app.core.body_limit import BodySizeLimitMiddleware
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.metrics import setup_metrics
from app.api.routes_quality import router as quality_router


def create_app() -> FastAPI:
    configure_logging(settings.log_json, settings.log_level)
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    # Added first, so it is the innermost middleware: its 413 responses still
    # get CORS headers, the request ID and Prometheus metrics.
    app.add_middleware(
        BodySizeLimitMiddleware, max_bytes=lambda: config.settings.max_request_bytes
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )
    app.include_router(quality_router)
    if settings.prometheus_enabled:
        setup_metrics(app)

    @app.get("/version")
    def version() -> dict:
        return {"app": settings.app_name, "version": settings.app_version}

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or f"rid-{uuid4()}"
        # Lets handlers put the same id in diagnostics and logs.
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    return app


app = create_app()
