import json
import asyncio
from contextlib import asynccontextmanager
from redis.asyncio import Redis, ConnectionPool

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from api.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.redis_pool = ConnectionPool.from_url(settings.redis_url)
    yield
    await app.redis_pool.aclose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


