from fastapi import Request
from fastapi.responses import JSONResponse


class ModelError(Exception):
    pass


async def model_exception_handler(request: Request, exc: ModelError):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )
