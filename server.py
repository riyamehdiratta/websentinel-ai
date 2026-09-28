#!/usr/bin/env python3
"""
WebSentinel AI - Intel® OpenVINO™ Fast REST API & Web Server
Provides high-speed endpoints for website health checks, visual defacement detection,
domain typosquatting risk analysis, and Intel hardware benchmarks.
"""

import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openvino as ov

from openvino_engine import OpenVINOEngine
from monitor_manager import MonitorManager

app = FastAPI(
    title="WebSentinel AI - Intel® OpenVINO™ Platform",
    description="Next-Gen Website Uptime & Visual Health Guardian powered by Intel OpenVINO",
    version="1.0.0"
)

# Enable CORS for local testing and web embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = OpenVINOEngine.get_instance()
monitor = MonitorManager()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -----------------
# Request Models
# -----------------
class AddSiteRequest(BaseModel):
    domain: str
    name: Optional[str] = None
    category: Optional[str] = "General"

class SimulateRequest(BaseModel):
    state: str  # healthy, error_500, error_404, defaced, cloudflare, maintenance

class DomainAnalyzeRequest(BaseModel):
    domain: str

# -----------------
# API Endpoints
# -----------------
@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>WebSentinel AI - Intel® OpenVINO™ Backend Ready</h1>")

@app.get("/api/health")
async def get_system_health():
    """Returns Intel OpenVINO runtime status and hardware device details."""
    return {
        "status": "online",
        "platform": "Intel® OpenVINO™ Toolkit",
        "openvino_version": ov.__version__,
        "primary_device": engine.primary_device,
        "available_devices": engine.available_devices,
        "device_info": engine.device_info,
        "models_loaded": {
            "visual_health_net": engine.compiled_visual is not None,
            "domain_risk_net": engine.compiled_domain is not None,
            "latency_anomaly_net": engine.compiled_latency is not None
        }
    }

@app.get("/api/sites")
async def get_sites():
    """Returns all monitored sites with current telemetry."""
    sites = monitor.get_all_sites()
    
    # Compute aggregate telemetry
    total = len(sites)
    operational = sum(1 for s in sites if s.get("overall_status") == "operational")
    threats = sum(1 for s in sites if s.get("overall_status") == "threat_detected")
    down = sum(1 for s in sites if s.get("overall_status") == "down")
    uptime_pct = round((operational / total * 100), 1) if total > 0 else 100.0

    return {
        "summary": {
            "total_monitored": total,
            "operational": operational,
            "threats_detected": threats,
            "down_nodes": down,
            "visual_uptime_pct": uptime_pct,
            "accelerator": f"Intel OpenVINO ({engine.primary_device})"
        },
        "sites": sites
    }

@app.post("/api/sites")
async def add_site(req: AddSiteRequest):
    """Adds a new domain to real-time monitoring."""
    if not req.domain or len(req.domain.strip()) < 3:
        raise HTTPException(status_code=400, detail="Invalid domain name")
    new_site = monitor.add_site(req.domain, req.name, req.category)
    return {"status": "success", "site": new_site}

@app.delete("/api/sites/{site_id}")
async def delete_site(site_id: str):
    """Deletes a site from monitoring."""
    success = monitor.delete_site(site_id)
    if not success:
        raise HTTPException(status_code=404, detail="Site ID not found")
    return {"status": "success", "message": f"Deleted {site_id}"}

@app.post("/api/sites/{site_id}/simulate")
async def simulate_site_state(site_id: str, req: SimulateRequest):
    """Simulates an outage, defacement, or recovery on a monitored site."""
    updated = monitor.update_site_simulation(site_id, req.state)
    if not updated:
        raise HTTPException(status_code=404, detail="Site not found")
    return {"status": "success", "site": updated}

@app.post("/api/sites/{site_id}/check")
async def check_single_site(site_id: str):
    """Triggers an instant OpenVINO audit for a specific site."""
    site = monitor.get_site(site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    checked = monitor.check_site_health(site)
    return {"status": "success", "site": checked}

@app.post("/api/sites/refresh-all")
async def refresh_all_sites():
    """Triggers a full cluster health audit across all domains."""
    updated = monitor.refresh_all()
    return {"status": "success", "total_sites": len(updated), "sites": updated}

@app.post("/api/domain/analyze")
async def analyze_domain_endpoint(req: DomainAnalyzeRequest):
    """Runs OpenVINO NLP brand armor and typosquatting scanner on any domain."""
    if not req.domain:
        raise HTTPException(status_code=400, detail="Domain is required")
    result = engine.analyze_domain_risk(req.domain)
    return {"status": "success", "analysis": result}

@app.get("/api/benchmark")
async def run_benchmark(iterations: int = 50):
    """Runs a live Intel OpenVINO inference hardware stress test and benchmark."""
    iters = min(200, max(10, iterations))
    benchmark_data = engine.run_hardware_benchmark(iterations=iters)
    return {"status": "success", "benchmark": benchmark_data}

if __name__ == "__main__":
    import uvicorn
    print("[Server] Starting WebSentinel AI on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
