"""FastAPI Application Entry Point for Synthetic Speech & DSP Lab."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api import routes_audio, routes_clone, routes_security
from backend.app.core.config import FRONTEND_DIR, HOST, PORT

app = FastAPI(
    title="AI Voice Cloning & DSP Acoustic Lab API",
    description="Full-stack platform for zero-shot voice cloning, signal processing analysis, and CNN deepfake voice detection.",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(routes_audio.router)
app.include_router(routes_clone.router)
app.include_router(routes_security.router)


@app.get("/api/health")
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "AI Voice Cloning & DSP Acoustic Lab",
        "version": "2.0.0"
    }


# Mount Frontend static files
dist_dir = FRONTEND_DIR / "dist"
static_target = dist_dir if dist_dir.exists() else FRONTEND_DIR

if static_target.exists():
    # Mount assets if dist exists
    assets_dir = static_target / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend_assets")

    @app.get("/")
    async def serve_index():
        """Serve frontend single-page application."""
        index_file = static_target / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Frontend index.html not found. Please run npm run build in frontend."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=HOST, port=PORT, reload=True)
