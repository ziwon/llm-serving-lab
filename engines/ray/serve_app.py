import os

import requests
from ray import serve
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse


@serve.deployment(route_prefix="/")
class OpenAIProxy:
    def __init__(self):
        self.backend_url = os.environ.get("RAY_BACKEND_URL", "http://vllm:8000").rstrip("/")

    async def __call__(self, request: Request):
        path = request.url.path
        method = request.method.lower()
        body = await request.body()
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "accept-encoding"}
        }
        target = f"{self.backend_url}{path}"

        if method == "post":
            upstream = requests.post(target, data=body, headers=headers, stream=True, timeout=600)
        elif method == "get":
            upstream = requests.get(target, headers=headers, stream=True, timeout=120)
        else:
            return Response(f"Unsupported method: {request.method}", status_code=405)

        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in {"content-encoding", "transfer-encoding", "connection"}
        }

        if "text/event-stream" in upstream.headers.get("content-type", ""):
            return StreamingResponse(
                upstream.iter_content(chunk_size=None),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type="text/event-stream",
            )

        return Response(
            upstream.content,
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )


app = OpenAIProxy.bind()
