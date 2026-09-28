# WebSentinel AI: Next-Gen Domain Manager & Visual Uptime Guardian
### Intelligent Website Monitoring, Real-Time Visual Defacement Detection, and Brand Protection

![FastAPI](https://img.shields.io/badge/FastAPI-Production%20REST-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Playwright](https://img.shields.io/badge/Playwright-Chromium%20Headless-2EAD33?style=for-the-badge&logo=playwright&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Intel OpenVINO](https://img.shields.io/badge/Intel%20OpenVINO-AI%20Acceleration-0068B5?style=for-the-badge&logo=intel&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Storage-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

---

## 🎯 The Real-World Problem

Traditional uptime monitoring tools (basic HTTP pingers) only test whether a web server responds with `HTTP 200 OK`. However, real-world web failures often occur above the transport layer:

- **Silent UI Defacement**: Compromised websites often continue serving `HTTP 200` while displaying a hacked banner or unauthorized content.
- **Broken SPAs & Blank Renders**: Modern JavaScript applications (React, Vue, Angular) can throw runtime exceptions or fail client-side chunk loading, serving a completely blank white screen while the server reports healthy status.
- **WAF & Cloudflare Blockades**: Monitoring bots and end-users can get stuck on interstitial CAPTCHA challenges or challenge pages without tripping traditional alerts.
- **Brand Impersonation & Phishing**: Threat actors register lookalike domains using homoglyphs (`inte1-login.com`, `paypa1-update.xyz`) and disposable TLDs to impersonate legitimate brands.
- **Silent SSL & DNS Degradation**: Overlooked TLS certificate expirations or missing DNS MX/SPF records quietly hurt deliverability and security.

**WebSentinel AI** is a comprehensive domain management and visual uptime platform. It combines **Headless Chromium screenshot captures**, **authoritative DNS resolution**, **live TLS/SSL socket handshakes**, and **embedded AI models** to verify that web applications are not only responding over the network, but are actually rendering properly and securely for users.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([DevOps / Security Admin]) -->|Add Domain or URL| API[FastAPI Server :8000]

    subgraph Probes ["Live Telemetry & Capture Engine"]
        API -->|Async Navigation| PW[Headless Chromium\nPlaywright Engine]
        PW -->|1280x800 Screenshot Buffer| Pixels[Raw Screenshot Buffer]
        
        API -->|Port 443 Socket| SSL_Probe[TLS Handshake Engine\nOpenSSL / SSLContext]
        SSL_Probe -->|x509 Certificate| SSL_Meta[Issuer, Cipher, TLSv1.3, Expiry]

        API -->|UDP/TCP Query| DNS_Probe[dnspython Resolver]
        DNS_Probe -->|Authoritative Records| DNS_Meta[A, AAAA, MX, NS, TXT, CNAME, SOA]
    end

    subgraph AI_Inference ["Accelerated AI Inference Engine"]
        Pixels -->|RGB Normalized Tensor| MB[visual_feature_net\nPyTorch MobileNetV2]
        Pixels -->|RGB Tensor [1, 3, 224, 224]| VH[visual_health_net\nVision Health Classifier]
        API -->|Domain Feature Vector [1, 32]| DR[domain_risk_net\nBrand Armor NLP Net]
        API -->|Rolling 20-Point Latency [1, 20]| LA[latency_anomaly_net\nPredictive Jitter Net]
    end

    VH -->|Visual Health Scores| Storage[(SQLite Database\nsentinel.db)]
    DR -->|Phishing & Typosquat Alerts| Storage
    LA -->|Latency Anomaly Alerts| Storage
    SSL_Meta --> Storage
    DNS_Meta --> Storage

    Storage --> Dashboard[WebSentinel Cyber Operations Dashboard]
```

---

## ⚡ Where & Why OpenVINO is Used

Running deep learning models alongside continuous uptime monitoring often introduces significant compute overhead and requires expensive GPU infrastructure. 

To solve this, **WebSentinel AI uses Intel® OpenVINO™ as the inference runtime** for its computer vision and NLP models. By compiling PyTorch and ONNX models into OpenVINO Intermediate Representation (`.xml` + `.bin`), the platform achieves:

- **Sub-Millisecond CPU Inference**: Vision models run in ~0.3ms to 1.2ms directly on standard CPUs without requiring discrete GPUs.
- **Resource Efficiency**: Negligible memory footprint, allowing high-frequency visual audits across large domain portfolios on standard commodity servers.
- **Hardware Portability**: Automatic hardware acceleration across standard CPUs, integrated GPUs, and dedicated neural accelerators.
- **Optimized Execution Paths**: Single-stream latency optimization (`PERFORMANCE_HINT: LATENCY`) for immediate screenshot checks, and throughput batching (`PERFORMANCE_HINT: THROUGHPUT`) for scanning domain fleets.

### Models Overview

| Model | Architecture | Role in WebSentinel | Latency |
| :--- | :--- | :--- | :--- |
| **`visual_feature_net`** | PyTorch MobileNetV2 | Extracts 1000-dimensional visual embeddings to detect unauthorized structural layout alterations over time. | **~1.2 ms** |
| **`visual_health_net`** | Multi-Layer Vision CNN | 5-Class Visual Uptime Classifier (Healthy, 404/500 Screen, Defaced, WAF/Cloudflare Block, Blank Render). | **~0.28 ms** |
| **`domain_risk_net`** | Character Embedding NLP Net | Evaluates homoglyph lookalikes (`0` vs `o`, `1` vs `l`), brand edit distance (Levenshtein), Shannon entropy, and high-risk TLDs. | **~0.15 ms** |
| **`latency_anomaly_net`** | Time-Series Predictive Net | Ingests rolling latency series to detect jitter patterns and predict impending outages before complete downtime occurs. | **~0.10 ms** |

---

## 🚀 Key Features

1. **Visual Uptime Guardian**:
   - Dispatches headless Chromium to capture high-resolution website screenshots.
   - Evaluates pixel render health with computer vision to distinguish between legitimate pages, blank SPA renders, Cloudflare challenges, and defacements.

2. **Domain Brand Armor & Typosquatting Hunter**:
   - Analyzes target domains for homoglyph deception, brand name edit distances, and disposable TLD patterns.
   - Flags suspicious phishing vectors targeting popular enterprise brands.

3. **Full-Stack Network Diagnostics**:
   - **Live DNS Inspector**: Authoritative queries for `A`, `AAAA`, `MX`, `NS`, `TXT`, `CNAME`, and `SOA` records with latency telemetry.
   - **Live SSL/TLS Auditor**: Real socket handshakes inspecting issuer, certificate validity range, TLS protocol version, cipher suite, and SAN domains.

4. **Interactive Cyber Operations Dashboard**:
   - Clean, modern operations center with real-time domain status, diagnostic modals, latency sparklines, and visual audit histories.

---

## 🛠️ Quickstart

### 1. Prerequisites
- Python 3.10+
- macOS, Linux, or Windows

### 2. Installation
```bash
git clone https://github.com/riyamehdiratta/websentinel-ai.git
cd websentinel-ai

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium
```

### 3. (Optional) Recompile OpenVINO Models
Pre-compiled models are already included in the `models/` directory. If you modify model architectures, rebuild them with:
```bash
python3 generate_models.py
```

### 4. Run Automated Test Suite
```bash
python3 -m unittest tests/test_pipeline.py
```

### 5. Launch the Platform
```bash
python3 server.py
```
Open **`http://localhost:8000`** in your browser to view the operations dashboard.

---

## 📡 API Overview

- `GET /api/health` — Platform status, hardware detection, and loaded neural models
- `GET /api/sites` — Monitored domain portfolio with visual health and latency metrics
- `POST /api/sites` — Add a new domain to monitor and trigger an immediate audit
- `POST /api/sites/{id}/check` — Trigger an instant live Playwright crawl + visual audit
- `GET /api/sites/{id}/dns` — Live authoritative DNS records and latency
- `GET /api/sites/{id}/ssl` — Live TLS handshake and certificate metadata
- `POST /api/domain/analyze` — Run brand armor NLP scanner on any domain
- `GET /api/benchmark` — Real-time AI engine throughput and latency performance check

---

## 📄 License
MIT License. Free for open source and enterprise monitoring.
