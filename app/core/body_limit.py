from typing import Callable

from fastapi import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

TOO_LARGE_DETAIL = "File too large"


class BodySizeLimitMiddleware:
    """
    Answers 413 for request bodies larger than max_bytes() without buffering
    them first: immediately when Content-Length declares too much, otherwise
    (chunked uploads) as soon as the bytes received pass the limit.
    """

    def __init__(self, app: ASGIApp, max_bytes: Callable[[], int]) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.max_bytes()
        declared = _content_length(scope)
        if declared is not None and declared > limit:
            response = JSONResponse({"detail": TOO_LARGE_DETAIL}, status_code=413)
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # Raised while FastAPI reads the body: it re-raises
                    # HTTPExceptions from there, Starlette closes the spooled
                    # upload files, and the client gets the usual 413 response.
                    raise HTTPException(status_code=413, detail=TOO_LARGE_DETAIL)
            return message

        await self.app(scope, limited_receive, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None  # the streamed byte count still applies
    return None
