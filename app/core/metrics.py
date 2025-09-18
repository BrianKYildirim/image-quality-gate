from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI


def setup_metrics(app: FastAPI) -> None:
    Instrumentator().instrument(app).expose(
        app, include_in_schema=False, should_gzip=True
    )
