#!/usr/bin/env python3
"""
WebSentinel AI - Intel® OpenVINO™ Production REST API Server
Provides high-throughput asynchronous endpoints for real-time website uptime,
deep visual defacement detection, DNS records, SSL certificates, and Intel hardware benchmarks.
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openvino as ov

from database import Database
from monitor_manager import MonitorManager
from dns_service import DNSService
from ssl_service import SSLService
from openvino_engine import OpenVINOEngine

app = FastAPI(
    title="WebSentinel AI - Intel® OpenVINO™ Production Platform",
    description="Real-world Website Uptime & Visual Health Guardian powered by Intel OpenVINO",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database.get_instance()
engine = OpenVINOEngine.get_instance()
monitor = MonitorManager.get_instance()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -----------------
# Request Models
# -----------------
class AddDomainRequest(BaseModel):
    domain: str
    name: Optional[str] = None
    category: Optional[str] = "General"

class DomainRiskRequest(BaseModel):
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
    """Returns Intel OpenVINO runtime status, available accelerators, and compiled models."""
    return {
        "status": "online",
        "platform": "Intel® OpenVINO™ Toolkit",
        "openvino_version": ov.__version__,
        "primary_device": engine.primary_device,
        "available_devices": engine.available_devices,
        "device_info": engine.device_info,
        "models_loaded": {
            "visual_feature_net (MobileNetV2)": engine.compiled_mobilenet is not None,
            "visual_health_net": engine.compiled_visual is not None,
            "domain_risk_net": engine.compiled_domain is not None,
            "latency_anomaly_net": engine.compiled_latency is not None
        }
    }

@app.get("/api/sites")
async def get_all_sites():
    """Returns all monitored domains with real-time audit telemetry and screenshots."""
    sites = monitor.get_all_sites()
    total = len(sites)
    operational = sum(1 for s in sites if s.get("overall_status") == "operational")
    threats = sum(1 for s in sites if s.get("overall_status") in ["threat_detected", "down", "waf_blocked"])
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
async def add_site(req: AddDomainRequest):
    """Adds any real internet domain and immediately triggers a live full-stack audit."""
    if not req.domain or len(req.domain.strip()) < 3:
        raise HTTPException(status_code=400, detail="Invalid domain name")
    
    try:
        new_site = monitor.add_site(req.domain, req.name, req.category)
        return {"status": "success", "site": new_site}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/sites/{domain_id}")
async def delete_site(domain_id: str):
    """Deletes a domain from portfolio monitoring."""
    success = monitor.delete_site(domain_id)
    if not success:
        raise HTTPException(status_code=404, detail="Domain ID not found")
    return {"status": "success", "message": f"Deleted {domain_id}"}

@app.get("/api/sites/{domain_id}/dns")
async def get_site_dns(domain_id: str):
    """Queries and returns real live DNS records for domain."""
    site = db.get_domain(domain_id)
    if not site:
        raise HTTPException(status_code=404, detail="Domain not found")
    
    dns_res = DNSService.resolve_all_records(site["domain"])
    db.save_dns_records(domain_id, dns_res["all_records"])
    return {"status": "success", "domain": site["domain"], "dns": dns_res}

@app.get("/api/sites/{domain_id}/ssl")
async def get_site_ssl(domain_id: str):
    """Performs a real TLS socket handshake and returns certificate telemetry."""
    site = db.get_domain(domain_id)
    if not site:
        raise HTTPException(status_code=404, detail="Domain not found")
    
    ssl_res = SSLService.inspect_ssl(site["domain"])
    db.save_ssl_cert(domain_id, ssl_res)
    return {"status": "success", "domain": site["domain"], "ssl": ssl_res}

@app.post("/api/sites/{domain_id}/check")
async def check_site_endpoint(domain_id: str):
    """Triggers an instant real Playwright crawl + OpenVINO neural audit for a specific site."""
    try:
        updated_site = await monitor.audit_single_domain(domain_id)
        return {"status": "success", "site": updated_site}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sites/refresh-all")
async def refresh_all_sites(background_tasks: BackgroundTasks):
    """Triggers real audit across all domains in background."""
    async def run_refresh():
        await monitor.audit_all_domains()

    background_tasks.add_task(run_refresh)
    return {"status": "success", "message": "Cluster audit dispatched across all domains"}

@app.post("/api/domain/analyze")
async def analyze_domain(req: DomainRiskRequest):
    """Runs OpenVINO NLP brand armor and typosquatting scanner on any domain."""
    if not req.domain:
        raise HTTPException(status_code=400, detail="Domain is required")
    analysis = engine.analyze_domain_risk(req.domain)
    return {"status": "success", "analysis": analysis}

@app.get("/api/benchmark")
async def run_hardware_benchmark(iterations: int = 50):
    """Runs live Intel OpenVINO hardware inference throughput & latency benchmark."""
    iters = min(200, max(10, iterations))
    benchmark = engine.run_hardware_benchmark(iterations=iters)
    return {"status": "success", "benchmark": benchmark}

if __name__ == "__main__":
    import uvicorn
    print("[Server] Launching WebSentinel AI Production Platform on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
