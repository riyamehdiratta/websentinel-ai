/**
 * WebSentinel AI - Intel® OpenVINO™ Client Application
 * Handles real-time telemetry, visual uptime watchdog, domain risk scanner,
 * fault injection simulations, and Intel hardware benchmarks.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  let sitesData = [];
  let currentFilter = 'all';
  let activeSimState = 'healthy';
  let selectedInspectorSite = null;

  // DOM Elements
  const sitesGrid = document.getElementById('sitesGrid');
  const metricUptime = document.getElementById('metricUptime');
  const metricTotalSites = document.getElementById('metricTotalSites');
  const pillOperational = document.getElementById('pillOperational');
  const pillThreats = document.getElementById('pillThreats');
  const metricSpeedup = document.getElementById('metricSpeedup');
  const intelDeviceLabel = document.getElementById('intelDeviceLabel');
  const avgInferenceLabel = document.getElementById('avgInferenceLabel');
  const benchmarkFpsBadge = document.getElementById('benchmarkFpsBadge');
  const toastContainer = document.getElementById('toastContainer');

  // Modal Elements
  const addSiteModal = document.getElementById('addSiteModal');
  const addSiteModalBtn = document.getElementById('addSiteModalBtn');
  const closeModalBtn = document.getElementById('closeModalBtn');
  const cancelModalBtn = document.getElementById('cancelModalBtn');
  const addSiteForm = document.getElementById('addSiteForm');
  const refreshAllBtn = document.getElementById('refreshAllBtn');
  const quickBenchmarkBtn = document.getElementById('quickBenchmarkBtn');

  // -------------------------------------------------------------
  // INITIALIZATION & TAB NAVIGATION
  // -------------------------------------------------------------
  function init() {
    setupTabs();
    setupEventListeners();
    fetchSystemHealth();
    fetchSites();
    runQuickBenchmarkOnLoad();
    
    // Auto-refresh every 12 seconds
    setInterval(fetchSites, 12000);
  }

  function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const targetTab = btn.getAttribute('data-tab');
        const targetPane = document.getElementById(targetTab);
        if (targetPane) {
          targetPane.classList.add('active');
          if (targetTab === 'inspectorTab') {
            updateInspectorView();
          } else if (targetTab === 'benchmarkTab') {
            executeBenchmark(30);
          }
        }
      });
    });
  }

  function setupEventListeners() {
    // Filter chips
    document.querySelectorAll('.filter-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        currentFilter = chip.getAttribute('data-filter');
        renderSites();
      });
    });

    // Refresh Fleet
    refreshAllBtn.addEventListener('click', async () => {
      showToast('Auditing entire domain fleet with Intel® OpenVINO™...', 'info');
      try {
        const res = await fetch('/api/sites/refresh-all', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
          sitesData = data.sites;
          renderSites();
          updateTelemetry({
            total_monitored: sitesData.length,
            operational: sitesData.filter(s => s.overall_status === 'operational').length,
            threats_detected: sitesData.filter(s => s.overall_status === 'threat_detected').length,
            down_nodes: sitesData.filter(s => s.overall_status === 'down').length,
            visual_uptime_pct: calculateUptimePct(sitesData)
          });
          showToast(`Audited ${data.total_sites} domains successfully`, 'success');
        }
      } catch (err) {
        showToast('Fleet audit failed: ' + err.message, 'error');
      }
    });

    // Modal
    addSiteModalBtn.addEventListener('click', () => addSiteModal.classList.add('active'));
    closeModalBtn.addEventListener('click', () => addSiteModal.classList.remove('active'));
    cancelModalBtn.addEventListener('click', () => addSiteModal.classList.remove('active'));
    
    addSiteForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const domain = document.getElementById('newDomainInput').value.trim();
      const name = document.getElementById('newNameInput').value.trim();
      const category = document.getElementById('newCategoryInput').value;

      try {
        const res = await fetch('/api/sites', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ domain, name, category })
        });
        const data = await res.json();
        if (data.status === 'success') {
          showToast(`Added ${domain} to OpenVINO Watchdog`, 'success');
          addSiteModal.classList.remove('active');
          addSiteForm.reset();
          fetchSites();
        } else {
          showToast(data.detail || 'Error adding domain', 'error');
        }
      } catch (err) {
        showToast('Failed to add site: ' + err.message, 'error');
      }
    });

    // Scanner
    const domainScanForm = document.getElementById('domainScanForm');
    domainScanForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const domain = document.getElementById('scanDomainInput').value.trim();
      runDomainAnalysis(domain);
    });

    document.querySelectorAll('.sample-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const d = btn.getAttribute('data-domain');
        document.getElementById('scanDomainInput').value = d;
        runDomainAnalysis(d);
      });
    });

    // Visual Inspector simulation controls
    document.querySelectorAll('.sim-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.sim-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeSimState = btn.getAttribute('data-state');
        runInspectorSimulation();
      });
    });

    document.getElementById('inspectorSiteSelect').addEventListener('change', (e) => {
      selectedInspectorSite = sitesData.find(s => s.id === e.target.value) || sitesData[0];
      runInspectorSimulation();
    });

    document.getElementById('runVisualInspectBtn').addEventListener('click', () => {
      runInspectorSimulation();
    });

    // Benchmark button
    quickBenchmarkBtn.addEventListener('click', () => {
      document.querySelector('[data-tab="benchmarkTab"]').click();
    });

    document.getElementById('startBenchmarkBtn').addEventListener('click', () => {
      const iters = parseInt(document.getElementById('benchmarkIters').value, 10);
      executeBenchmark(iters);
    });
  }

  // -------------------------------------------------------------
  // API DATA FETCHING
  // -------------------------------------------------------------
  async function fetchSystemHealth() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      if (data.status === 'online') {
        const dev = data.primary_device;
        const devFull = data.device_info[dev] || dev;
        intelDeviceLabel.textContent = `${dev} (${devFull})`;
      }
    } catch (err) {
      console.error('Error fetching health:', err);
    }
  }

  async function fetchSites() {
    try {
      const res = await fetch('/api/sites');
      const data = await res.json();
      sitesData = data.sites;
      updateTelemetry(data.summary);
      renderSites();
      populateInspectorSelect();
    } catch (err) {
      console.error('Error fetching sites:', err);
    }
  }

  function calculateUptimePct(sites) {
    if (!sites || sites.length === 0) return 100.0;
    const op = sites.filter(s => s.overall_status === 'operational').length;
    return Math.round((op / sites.length) * 1000) / 10;
  }

  function updateTelemetry(summary) {
    if (!summary) return;
    metricUptime.textContent = `${summary.visual_uptime_pct}%`;
    metricTotalSites.textContent = summary.total_monitored;
    pillOperational.textContent = `${summary.operational} Operational`;
    pillThreats.textContent = `${summary.threats_detected + summary.down_nodes} Issues`;
    
    document.getElementById('countAll').textContent = summary.total_monitored;
  }

  // -------------------------------------------------------------
  // RENDERING SITE CARDS
  // -------------------------------------------------------------
  function renderSites() {
    if (!sitesGrid) return;
    
    let filtered = sitesData;
    if (currentFilter === 'operational') {
      filtered = sitesData.filter(s => s.overall_status === 'operational');
    } else if (currentFilter === 'threat') {
      filtered = sitesData.filter(s => s.overall_status !== 'operational');
    }

    if (filtered.length === 0) {
      sitesGrid.innerHTML = `
        <div class="loading-state">
          <p>No sites match the selected filter.</p>
        </div>
      `;
      return;
    }

    sitesGrid.innerHTML = filtered.map(site => {
      const vHealth = site.visual_health || {};
      const dRisk = site.domain_risk || {};
      const status = site.overall_status || 'operational';
      const isThreat = status !== 'operational';
      
      const statusClass = status;
      const statusLabel = status.replace('_', ' ');

      // Sparkline HTML
      const latHistory = site.latency_history || [120, 110, 115];
      const maxLat = Math.max(...latHistory, 300);
      const sparkBars = latHistory.slice(-12).map(l => {
        const heightPct = Math.min(100, Math.max(15, (l / maxLat) * 100));
        const isSpike = l > 500;
        return `<div class="spark-tick ${isSpike ? 'spike' : ''}" style="height: ${heightPct}%" title="${l}ms"></div>`;
      }).join('');

      return `
        <div class="site-card ${isThreat ? 'threat-border' : ''}" id="card_${site.id}">
          <div class="site-card-header">
            <div class="site-title-box">
              <div class="site-name">${escapeHtml(site.name)}</div>
              <a href="${site.url}" target="_blank" class="site-domain-link">
                ${escapeHtml(site.domain)}
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                  <polyline points="15 3 21 3 21 9"></polyline>
                  <line x1="10" y1="14" x2="21" y2="3"></line>
                </svg>
              </a>
            </div>
            <div class="site-status-badge ${statusClass}">
              <span class="pulse-dot" style="background-color: ${statusClass === 'operational' ? '#10B981' : '#EF4444'}"></span>
              ${statusLabel}
            </div>
          </div>

          <div class="site-snapshot-wrap">
            <img class="site-snapshot-img" src="${site.snapshot_preview || ''}" alt="Render Preview">
            <div class="snapshot-ov-overlay">
              <span class="ov-icon">⚡ OpenVINO Vision:</span>
              <span>${vHealth.label || 'Healthy'} (${vHealth.confidence || 98}%)</span>
            </div>
          </div>

          <div class="site-body">
            <div class="site-indicators-grid">
              <div class="indicator-box">
                <div class="indicator-label">HTTP Ping</div>
                <div class="indicator-val ${site.http_status === 200 ? 'green' : 'red'}">${site.http_status} OK</div>
              </div>
              <div class="indicator-box">
                <div class="indicator-label">Response Time</div>
                <div class="indicator-val ${site.http_latency_ms < 300 ? 'green' : 'amber'}">${site.http_latency_ms}ms</div>
              </div>
              <div class="indicator-box">
                <div class="indicator-label">Brand Risk</div>
                <div class="indicator-val ${dRisk.risk_score > 50 ? 'red' : 'green'}">${dRisk.risk_score || 0}%</div>
              </div>
            </div>

            <div class="latency-spark-row">
              <div class="indicator-label">Latency Jitter (OpenVINO Anomaly Net)</div>
              <div class="spark-bars">${sparkBars}</div>
            </div>

            <!-- Fault Injection Simulator in Card -->
            <div class="simulation-bar">
              <div class="sim-label-row">
                <span>Inject Fault & Test OpenVINO:</span>
                <span style="color: var(--intel-cyan)">Live State: ${site.simulated_state}</span>
              </div>
              <div class="sim-btn-row">
                <button class="sim-chip ${site.simulated_state === 'healthy' ? 'active' : ''}" onclick="simulateState('${site.id}', 'healthy')">Healthy</button>
                <button class="sim-chip ${site.simulated_state === 'error_500' ? 'active' : ''}" onclick="simulateState('${site.id}', 'error_500')">500 Err</button>
                <button class="sim-chip ${site.simulated_state === 'defaced' ? 'active' : ''}" onclick="simulateState('${site.id}', 'defaced')">Defaced</button>
                <button class="sim-chip ${site.simulated_state === 'cloudflare' ? 'active' : ''}" onclick="simulateState('${site.id}', 'cloudflare')">WAF</button>
              </div>
            </div>
          </div>

          <div class="site-card-footer">
            <div class="audit-time">Audited in ${vHealth.inference_time_ms || '0.3'}ms on OpenVINO CPU</div>
            <div class="card-actions">
              <button class="btn-card-action" onclick="checkSite('${site.id}')" title="Recheck Site">⚡ Recheck</button>
              <button class="btn-card-action danger" onclick="deleteSite('${site.id}')" title="Delete">✕</button>
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  // -------------------------------------------------------------
  // SIMULATION & ACTIONS
  // -------------------------------------------------------------
  window.simulateState = async function(siteId, state) {
    showToast(`Injecting simulated state '${state}' into OpenVINO model...`, 'info');
    try {
      const res = await fetch(`/api/sites/${siteId}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state })
      });
      const data = await res.json();
      if (data.status === 'success') {
        const idx = sitesData.findIndex(s => s.id === siteId);
        if (idx !== -1) {
          sitesData[idx] = data.site;
          renderSites();
          updateTelemetry({
            total_monitored: sitesData.length,
            operational: sitesData.filter(s => s.overall_status === 'operational').length,
            threats_detected: sitesData.filter(s => s.overall_status === 'threat_detected').length,
            down_nodes: sitesData.filter(s => s.overall_status === 'down').length,
            visual_uptime_pct: calculateUptimePct(sitesData)
          });
          showToast(`OpenVINO re-evaluated site state: ${data.site.visual_health.label}`, 'success');
        }
      }
    } catch (err) {
      showToast('Simulation update failed: ' + err.message, 'error');
    }
  };

  window.checkSite = async function(siteId) {
    showToast('Running OpenVINO neural audit...', 'info');
    try {
      const res = await fetch(`/api/sites/${siteId}/check`, { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        const idx = sitesData.findIndex(s => s.id === siteId);
        if (idx !== -1) {
          sitesData[idx] = data.site;
          renderSites();
          showToast('Audit complete in ' + data.site.total_audit_ms + 'ms', 'success');
        }
      }
    } catch (err) {
      showToast('Check failed: ' + err.message, 'error');
    }
  };

  window.deleteSite = async function(siteId) {
    if (!confirm('Stop monitoring this domain?')) return;
    try {
      const res = await fetch(`/api/sites/${siteId}`, { method: 'DELETE' });
      const data = await res.json();
      if (data.status === 'success') {
        showToast('Site removed', 'info');
        fetchSites();
      }
    } catch (err) {
      showToast('Delete failed: ' + err.message, 'error');
    }
  };

  // -------------------------------------------------------------
  // BRAND ARMOR & DOMAIN SCANNER
  // -------------------------------------------------------------
  async function runDomainAnalysis(domain) {
    const scannerResults = document.getElementById('scannerResults');
    const scanSubmitBtn = document.getElementById('scanSubmitBtn');
    
    scanSubmitBtn.disabled = true;
    scannerResults.innerHTML = `
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Analyzing character embeddings with OpenVINO DomainRiskNet...</p>
      </div>
    `;

    try {
      const res = await fetch('/api/domain/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domain })
      });
      const data = await res.json();
      scanSubmitBtn.disabled = false;

      if (data.status === 'success') {
        renderDomainAnalysisResult(data.analysis);
      } else {
        scannerResults.innerHTML = `<div class="empty-state"><h3>Analysis failed</h3></div>`;
      }
    } catch (err) {
      scanSubmitBtn.disabled = false;
      showToast('Domain scan error: ' + err.message, 'error');
    }
  }

  function renderDomainAnalysisResult(res) {
    const scannerResults = document.getElementById('scannerResults');
    const isSafe = res.risk_level === 'low';
    const scoreColor = isSafe ? '#10B981' : (res.risk_level === 'medium' ? '#F59E0B' : '#EF4444');

    const threatListHtml = (res.threat_vectors || []).map(tv => `
      <div class="threat-item ${tv.severity}">
        <div>
          <div class="threat-title">[${tv.severity}] ${tv.type}</div>
          <div class="threat-detail">${escapeHtml(tv.detail)}</div>
        </div>
      </div>
    `).join('') || `<div class="threat-item" style="border-left-color: #10B981"><div class="threat-title">Authentic Domain Structure</div><div class="threat-detail">No typosquatting, homoglyph deception, or suspicious TLD patterns found.</div></div>`;

    scannerResults.innerHTML = `
      <div class="section-badge ${isSafe ? 'blue' : 'purple'}">OpenVINO Classification Result</div>
      <h3 style="font-size: 1.25rem; font-weight: 800; color: #fff; margin-bottom: 12px;">${escapeHtml(res.domain)}</h3>

      <div class="risk-score-display">
        <div>
          <div style="font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase;">Neural Threat Score</div>
          <div class="risk-score-num" style="color: ${scoreColor}">${res.risk_score} <span style="font-size: 1.1rem; color: var(--text-muted)">/ 100</span></div>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 0.8rem; color: var(--text-secondary)">Category</div>
          <div style="font-size: 1rem; font-weight: 700; color: #fff">${res.risk_label}</div>
          <div style="font-size: 0.75rem; color: var(--intel-cyan); font-family: var(--font-mono)">Confidence: ${res.confidence}%</div>
        </div>
      </div>

      <div style="display: flex; gap: 10px; margin-bottom: 16px;">
        <div class="indicator-box" style="flex: 1">
          <div class="indicator-label">Inference Latency</div>
          <div class="indicator-val green">${res.inference_time_ms} ms</div>
        </div>
        <div class="indicator-box" style="flex: 1">
          <div class="indicator-label">Lexical Entropy</div>
          <div class="indicator-val">${res.entropy} bits</div>
        </div>
        <div class="indicator-box" style="flex: 1">
          <div class="indicator-label">Target Brand</div>
          <div class="indicator-val ${res.target_brand_detected ? 'red' : 'green'}">${res.target_brand_detected ? res.target_brand_detected.toUpperCase() : 'None (Organic)'}</div>
        </div>
      </div>

      <div style="font-size: 0.85rem; font-weight: 700; color: #fff; margin-top: 14px;">Threat Vector Breakdown:</div>
      <div class="threat-list">
        ${threatListHtml}
      </div>
    `;
  }

  // -------------------------------------------------------------
  // VISUAL DEFACEMENT INSPECTOR TAB
  // -------------------------------------------------------------
  function populateInspectorSelect() {
    const sel = document.getElementById('inspectorSiteSelect');
    if (!sel) return;
    sel.innerHTML = sitesData.map(s => `
      <option value="${s.id}">${escapeHtml(s.name)} (${s.domain})</option>
    `).join('');
    if (!selectedInspectorSite && sitesData.length > 0) {
      selectedInspectorSite = sitesData[0];
    }
  }

  function updateInspectorView() {
    populateInspectorSelect();
    runInspectorSimulation();
  }

  async function runInspectorSimulation() {
    const selSite = selectedInspectorSite || sitesData[0];
    if (!selSite) return;

    // Simulate state update on backend
    try {
      const res = await fetch(`/api/sites/${selSite.id}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: activeSimState })
      });
      const data = await res.json();
      if (data.status === 'success') {
        const site = data.site;
        const vh = site.visual_health;
        
        // Update snapshot preview
        document.getElementById('inspectSnapshotImg').src = site.snapshot_preview;
        document.getElementById('inspectDeviceTag').textContent = `Runtime: Intel® OpenVINO™ (${vh.device})`;

        // Render detailed class breakdown
        const breakdownBars = (vh.breakdown || []).map(b => `
          <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 600; margin-bottom: 3px;">
              <span style="color: #fff">${b.label}</span>
              <span style="color: ${b.color}; font-family: var(--font-mono)">${b.probability}%</span>
            </div>
            <div style="width: 100%; height: 6px; background: rgba(30, 41, 59, 0.6); border-radius: 3px; overflow: hidden;">
              <div style="width: ${b.probability}%; height: 100%; background: ${b.color}; transition: width 0.3s ease;"></div>
            </div>
          </div>
        `).join('');

        document.getElementById('inspectResultsCard').innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
              <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase;">Visual Diagnosis</div>
              <div style="font-size: 1.15rem; font-weight: 800; color: #fff;">${vh.label}</div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 0.75rem; color: var(--text-secondary);">Inference Time</div>
              <div style="font-size: 1rem; font-weight: 800; color: var(--intel-cyan); font-family: var(--font-mono)">${vh.inference_time_ms} ms</div>
            </div>
          </div>
          <div style="margin-top: 14px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); margin-bottom: 10px;">Neural Class Probabilities (OpenVINO Vision CNN):</div>
            ${breakdownBars}
          </div>
        `;
      }
    } catch (err) {
      console.error('Inspector error:', err);
    }
  }

  // -------------------------------------------------------------
  // BENCHMARK SUITE
  // -------------------------------------------------------------
  async function runQuickBenchmarkOnLoad() {
    try {
      const res = await fetch('/api/benchmark?iterations=25');
      const data = await res.json();
      if (data.status === 'success') {
        const bm = data.benchmark;
        avgInferenceLabel.textContent = `${bm.latency_p50_ms} ms`;
        benchmarkFpsBadge.textContent = `${bm.throughput_fps} FPS`;
        metricSpeedup.textContent = `${bm.speedup_vs_unoptimized} Speedup`;
        document.getElementById('metricHwDetails').textContent = `${bm.device_full_name} (${bm.device})`;
      }
    } catch (e) {
      console.error(e);
    }
  }

  async function executeBenchmark(iterations) {
    const grid = document.getElementById('benchmarkResultsGrid');
    const startBtn = document.getElementById('startBenchmarkBtn');
    
    startBtn.disabled = true;
    grid.innerHTML = `
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Executing ${iterations} OpenVINO inference loops on ${intelDeviceLabel.textContent}...</p>
      </div>
    `;

    try {
      const res = await fetch(`/api/benchmark?iterations=${iterations}`);
      const data = await res.json();
      startBtn.disabled = false;

      if (data.status === 'success') {
        const bm = data.benchmark;
        avgInferenceLabel.textContent = `${bm.latency_p50_ms} ms`;
        benchmarkFpsBadge.textContent = `${bm.throughput_fps} FPS`;
        metricSpeedup.textContent = `${bm.speedup_vs_unoptimized} Speedup`;

        grid.innerHTML = `
          <div class="bm-metric-box">
            <div class="bm-metric-label">Inference Throughput</div>
            <div class="bm-metric-value cyan">${bm.throughput_fps}</div>
            <div class="bm-metric-sub">Frames / Inferences Per Second</div>
          </div>
          <div class="bm-metric-box">
            <div class="bm-metric-label">Median Latency (P50)</div>
            <div class="bm-metric-value green">${bm.latency_p50_ms} ms</div>
            <div class="bm-metric-sub">Ultra-low latency inference</div>
          </div>
          <div class="bm-metric-box">
            <div class="bm-metric-label">OpenVINO Acceleration</div>
            <div class="bm-metric-value" style="color: #A855F7">${bm.speedup_vs_unoptimized}</div>
            <div class="bm-metric-sub">Speedup vs standard Python loop</div>
          </div>
          <div class="bm-metric-box">
            <div class="bm-metric-label">Tail Latency (P95 / P99)</div>
            <div class="bm-metric-value" style="font-size: 1.5rem">${bm.latency_p95_ms} / ${bm.latency_p99_ms} ms</div>
            <div class="bm-metric-sub">Predictable performance under load</div>
          </div>
          <div class="bm-metric-box">
            <div class="bm-metric-label">Hardware Device Target</div>
            <div class="bm-metric-value" style="font-size: 1.4rem; color: #38BDF8">${bm.device_full_name}</div>
            <div class="bm-metric-sub">Architecture: ${bm.device} (Optimized with OpenVINO ${bm.openvino_version.split('-')[0]})</div>
          </div>
          <div class="bm-metric-box">
            <div class="bm-metric-label">Total Test Samples</div>
            <div class="bm-metric-value">${bm.total_iterations} Runs</div>
            <div class="bm-metric-sub">Average: ${bm.latency_avg_ms}ms | Min: ${bm.latency_min_ms}ms</div>
          </div>
        `;
        showToast('Intel® OpenVINO™ Benchmark Complete', 'success');
      }
    } catch (err) {
      startBtn.disabled = false;
      showToast('Benchmark error: ' + err.message, 'error');
    }
  }

  // -------------------------------------------------------------
  // TOAST NOTIFICATIONS & UTILITIES
  // -------------------------------------------------------------
  function showToast(message, type = 'info') {
    if (!toastContainer) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '⚠️';

    toast.innerHTML = `<span>${icon}</span> <span>${escapeHtml(message)}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.toString()
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  init();
});
