"""
Streaming Transaction Replay and WebSocket Feed Simulator (Sprint 5).

Runs an asynchronous generator replaying transactions and triggering real-time AML alerts.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Dict, Any, List, Set
from fastapi import WebSocket


class StreamingTransactionSimulator:
    """
    Manages active WebSocket connections and broadcasts real-time alerts.
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.is_running: bool = False
        self.interval_seconds: float = 1.0

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(alert_data)
            except Exception:
                disconnected.add(connection)
        for dead in disconnected:
            self.active_connections.discard(dead)

    async def run_simulation_loop(self, sample_alerts: List[Dict[str, Any]]):
        self.is_running = True
        idx = 0
        n = len(sample_alerts)
        while self.is_running and n > 0:
            alert = sample_alerts[idx % n].copy()
            alert["timestamp"] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            alert["stream_event_id"] = f"EVT_{datetime.datetime.utcnow().timestamp():.0f}"
            await self.broadcast_alert(alert)
            idx += 1
            await asyncio.sleep(self.interval_seconds)


streaming_simulator = StreamingTransactionSimulator()
