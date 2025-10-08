from fastapi import FastAPI
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}",
            "version": settings.VERSION,
            "docs": "/docs"
    }
    
@app.get("/api/v1/provinces")
async def get_provinces():
    return {"provinces": list(settings.PROVINCES.values())}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}