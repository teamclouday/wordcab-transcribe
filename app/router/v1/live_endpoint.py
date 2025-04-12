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
"""Live endpoints for the Wordcab Transcribe API."""

import shortuuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.dependencies import asr_live

router = APIRouter()

BUFFER_SIZE = 4096 * 4
BUFFER_OVERLAP_SIZE = 1024


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self.active_connections: dict[str, WebSocket] = {}
        self.audio_buffers: dict[str, bytes] = {}

    async def connect(self, sid: str, websocket: WebSocket) -> None:
        """Connect a WebSocket."""
        if len(self.active_connections) > 1:
            await websocket.close(code=1001, reason="Too many connections, try again later.")

        else:
            await websocket.accept()
            self.active_connections[sid] = websocket
            self.audio_buffers[sid] = b""

    def disconnect(self, sid: str) -> None:
        """Disconnect a WebSocket."""
        self.active_connections.pop(sid, None)
        self.audio_buffers.pop(sid, None)


manager = ConnectionManager()


@router.websocket("")
async def websocket_endpoint(source_lang: str, websocket: WebSocket) -> None:
    """Handle WebSocket connections."""
    sid = shortuuid.uuid()
    await manager.connect(sid, websocket)

    try:
        while True:
            data = await websocket.receive_bytes()

            manager.audio_buffers[sid] += data

            if len(manager.audio_buffers[sid]) > BUFFER_SIZE:
                data_to_process = manager.audio_buffers[sid]
                # keep some overlap for context
                manager.audio_buffers[sid] = data_to_process[-BUFFER_OVERLAP_SIZE:]

                async for result in asr_live.process_input(sid=sid, data=data_to_process, source_lang=source_lang):
                    await websocket.send_json(result)
                    del result

    except WebSocketDisconnect:
        manager.disconnect(sid)

    finally:
        asr_live.clean_up_states(sid)
