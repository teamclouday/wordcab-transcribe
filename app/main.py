# Copyright 2024 The Wordcab Team. All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Wordcab/wordcab-transcribe/blob/main/LICENSE
#
# Except as expressly provided otherwise herein, and to the fullest
# extent permitted by law, Licensor provides the Software (and each
# Contributor provides its Contributions) AS IS, and Licensor
# disclaims all warranties or guarantees of any kind, express or
# implied, whether arising under any law or from any usage in trade,
# or otherwise including but not limited to the implied warranties
# of merchantability, non-infringement, quiet enjoyment, fitness
# for a particular purpose, or otherwise.
#
# See the License for the specific language governing permissions
# and limitations under the License.

"""Main API module of the Wordcab Transcribe."""

from fastapi import Depends, FastAPI
from fastapi import status as http_status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import settings
from app.dependencies import lifespan
from app.logging import LoggingMiddleware
from app.router.authentication import get_current_user
from app.router.v1.endpoints import (
    api_router,
    auth_router,
)

# Main application instance creation
app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    debug=settings.debug,
    lifespan=lifespan,
)

# Add logging middleware
app.add_middleware(LoggingMiddleware, debug_mode=settings.debug)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the appropriate routers based on the settings
if settings.debug is False:
    app.include_router(auth_router, tags=["authentication"])
    app.include_router(api_router, prefix=settings.api_prefix, dependencies=[Depends(get_current_user)])
else:
    app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/", tags=["status"])
async def home() -> HTMLResponse:
    """Root endpoint returning a simple HTML page with the project info."""
    content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{settings.project_name}</title>
        <link href="https://unpkg.com/tailwindcss@^2/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="h-screen mx-auto text-center flex min-h-full items-center justify-center bg-gray-100 text-gray-700">
        <div class="container mx-auto p-4">
            <h1 class="text-4xl font-medium">{settings.project_name}</h1>
            <p class="text-gray-600">Version: {settings.version}</p>
            <p class="text-gray-600">{settings.description}</p>
            <a href="/docs">
                <button class="mt-8 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Docs</button>
            </a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=content, media_type="text/html")


@app.get("/healthz", status_code=http_status.HTTP_200_OK, tags=["status"])
async def health() -> dict:
    """Health check endpoint. Important for Kubernetes liveness probe."""
    return {"status": "ok"}
