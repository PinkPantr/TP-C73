import os
from contextlib import asynccontextmanager

import h2o
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    h2o.connect(
        url=os.environ["H2O_URL"],
        strict_version_check=True,
    )
    yield
    h2o.connection().close()


app = FastAPI(
    title="Consomation de carburant au Canada pour les vehicules des années 2015-2024",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}