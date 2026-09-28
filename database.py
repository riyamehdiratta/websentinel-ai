#!/usr/bin/env python3
"""
WebSentinel AI - Production Database Layer (SQLite)
Handles persistent storage for domain portfolios, audit telemetry, DNS records, and SSL certs.
"""

import os
import sqlite3
import json
import time
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "sentinel.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

class Database:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Database()
        return cls._instance

    def __init__(self):
        self._init_schema()
        self._seed_default_domains()

    def get_connection(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Domains table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS domains (
                id TEXT PRIMARY KEY,
                domain TEXT UNIQUE NOT NULL,
                url TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT DEFAULT 'General',
                created_at REAL NOT NULL,
                last_checked REAL,
                is_monitored INTEGER DEFAULT 1
            )
            """)

            # Audit History table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                http_status INTEGER,
                http_latency_ms REAL,
                dns_latency_ms REAL,
                visual_health_class TEXT,
                visual_health_label TEXT,
                visual_confidence REAL,
                health_score INTEGER,
                screenshot_path TEXT,
                domain_risk_level TEXT,
                domain_risk_score INTEGER,
                overall_status TEXT,
                audit_meta_json TEXT,
                FOREIGN KEY (domain_id) REFERENCES domains(id) ON DELETE CASCADE
            )
            """)

            # DNS Records Cache
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS dns_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain_id TEXT NOT NULL,
                record_type TEXT NOT NULL,
                value TEXT NOT NULL,
                ttl INTEGER,
                last_updated REAL NOT NULL,
                FOREIGN KEY (domain_id) REFERENCES domains(id) ON DELETE CASCADE
            )
            """)

            # SSL Certificates table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS ssl_certs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain_id TEXT UNIQUE NOT NULL,
                issuer TEXT,
                subject_cn TEXT,
                valid_from TEXT,
                valid_to TEXT,
                days_remaining INTEGER,
                cipher_suite TEXT,
                tls_version TEXT,
                san_list_json TEXT,
                last_checked REAL NOT NULL,
                FOREIGN KEY (domain_id) REFERENCES domains(id) ON DELETE CASCADE
            )
            """)
            conn.commit()

    def _seed_default_domains(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM domains")
            if cursor.fetchone()[0] == 0:
                defaults = [
                    ("site_intel", "intel.com", "https://www.intel.com", "Intel Corporation", "Hardware & AI Infrastructure"),
                    ("site_github", "github.com", "https://github.com", "GitHub Developer Platform", "Developer Cloud"),
                    ("site_python", "python.org", "https://www.python.org", "Python Software Foundation", "Open Source Language"),
                    ("site_wikipedia", "wikipedia.org", "https://www.wikipedia.org", "Wikipedia Knowledge Base", "Knowledge & Reference")
                ]
                now = time.time()
                for d_id, dom, url, name, cat in defaults:
                    cursor.execute(
                        "INSERT INTO domains (id, domain, url, name, category, created_at, last_checked) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (d_id, dom, url, name, cat, now, now)
                    )
                conn.commit()

    def get_all_domains(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM domains ORDER BY created_at ASC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_domain(self, domain_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM domains WHERE id = ?", (domain_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_domain(self, domain_id: str, domain: str, url: str, name: str, category: str) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = time.time()
            cursor.execute(
                "INSERT OR REPLACE INTO domains (id, domain, url, name, category, created_at, last_checked) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (domain_id, domain, url, name, category, now, now)
            )
            conn.commit()
            return self.get_domain(domain_id)

    def delete_domain(self, domain_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM domains WHERE id = ?", (domain_id,))
            conn.commit()
            return cursor.rowcount > 0

    def record_audit(self, audit_data: Dict[str, Any]) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO health_audits (
                domain_id, timestamp, http_status, http_latency_ms, dns_latency_ms,
                visual_health_class, visual_health_label, visual_confidence,
                health_score, screenshot_path, domain_risk_level, domain_risk_score,
                overall_status, audit_meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_data["domain_id"],
                audit_data.get("timestamp", time.time()),
                audit_data.get("http_status", 200),
                audit_data.get("http_latency_ms", 0.0),
                audit_data.get("dns_latency_ms", 0.0),
                audit_data.get("visual_health_class", "HEALTHY_OPERATIONAL"),
                audit_data.get("visual_health_label", "Healthy"),
                audit_data.get("visual_confidence", 95.0),
                audit_data.get("health_score", 100),
                audit_data.get("screenshot_path", ""),
                audit_data.get("domain_risk_level", "low"),
                audit_data.get("domain_risk_score", 0),
                audit_data.get("overall_status", "operational"),
                json.dumps(audit_data.get("meta", {}))
            ))
            
            # Update last_checked in domains
            cursor.execute("UPDATE domains SET last_checked = ? WHERE id = ?", (time.time(), audit_data["domain_id"]))
            conn.commit()
            return cursor.lastrowid

    def get_latest_audit(self, domain_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM health_audits WHERE domain_id = ? ORDER BY timestamp DESC LIMIT 1",
                (domain_id,)
            )
            row = cursor.fetchone()
            if row:
                d = dict(row)
                if d.get("audit_meta_json"):
                    try:
                        d["meta"] = json.loads(d["audit_meta_json"])
                    except Exception:
                        d["meta"] = {}
                return d
            return None

    def get_audit_history(self, domain_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM health_audits WHERE domain_id = ? ORDER BY timestamp ASC LIMIT ?",
                (domain_id, limit)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def save_dns_records(self, domain_id: str, records: List[Dict[str, Any]]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dns_records WHERE domain_id = ?", (domain_id,))
            now = time.time()
            for r in records:
                cursor.execute(
                    "INSERT INTO dns_records (domain_id, record_type, value, ttl, last_updated) VALUES (?, ?, ?, ?, ?)",
                    (domain_id, r["type"], r["value"], r.get("ttl", 300), now)
                )
            conn.commit()

    def get_dns_records(self, domain_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM dns_records WHERE domain_id = ?", (domain_id,))
            return [dict(r) for r in cursor.fetchall()]

    def save_ssl_cert(self, domain_id: str, cert_info: Dict[str, Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO ssl_certs (
                domain_id, issuer, subject_cn, valid_from, valid_to, days_remaining,
                cipher_suite, tls_version, san_list_json, last_checked
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                domain_id,
                cert_info.get("issuer", ""),
                cert_info.get("subject_cn", ""),
                cert_info.get("valid_from", ""),
                cert_info.get("valid_to", ""),
                cert_info.get("days_remaining", 0),
                cert_info.get("cipher_suite", ""),
                cert_info.get("tls_version", ""),
                json.dumps(cert_info.get("san_list", [])),
                time.time()
            ))
            conn.commit()

    def get_ssl_cert(self, domain_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ssl_certs WHERE domain_id = ?", (domain_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                if d.get("san_list_json"):
                    try:
                        d["san_list"] = json.loads(d["san_list_json"])
                    except Exception:
                        d["san_list"] = []
                return d
            return None

if __name__ == "__main__":
    db = Database.get_instance()
    domains = db.get_all_domains()
    print(f"Database initialized with {len(domains)} seeded domains:")
    for d in domains:
        print(f"- {d['name']} ({d['domain']})")
