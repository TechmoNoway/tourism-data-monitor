from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import router as api_router
from app.database.connection import create_tables

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for monitoring tourism data from social platforms"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    """Khởi tạo database khi app start"""
    create_tables()
    print(f"🚀 {settings.PROJECT_NAME} started successfully!")
    print("📍 Server: http://127.0.0.1:4242")
    print("📖 Docs: http://127.0.0.1:4242/docs")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs": "/docs",
        "api_endpoints": {
            "provinces": f"{settings.API_V1_STR}/provinces",
            "attractions": f"{settings.API_V1_STR}/attractions",
            "health": f"{settings.API_V1_STR}/health"
        }
    }