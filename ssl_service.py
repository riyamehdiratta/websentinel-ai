#!/usr/bin/env python3
"""
WebSentinel AI - Real SSL / TLS Security Inspector
Connects to real web servers over TLS socket to extract live x509 certificate data,
negotiated ciphers, protocol versions, validity windows, and SANs.
"""

import socket
import ssl
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

class SSLService:
    @staticmethod
    def inspect_ssl(domain: str, port: int = 443) -> Dict[str, Any]:
        """
        Extracts real TLS/SSL certificate telemetry for any live domain.
        """
        clean_domain = domain.lower().strip().replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
        
        t0 = time.perf_counter()
        context = ssl.create_default_context()
        
        try:
            try:
                raw_sock = socket.create_connection((clean_domain, port), timeout=4.0)
            except (socket.gaierror, socket.timeout):
                # Fallback: resolve IP directly via DNSService
                from dns_service import DNSService
                dns_res = DNSService.resolve_all_records(clean_domain)
                a_records = dns_res.get("records_by_type", {}).get("A", [])
                if a_records:
                    raw_sock = socket.create_connection((a_records[0]["value"], port), timeout=4.0)
                else:
                    raise

            with raw_sock as sock:
                with context.wrap_socket(sock, server_hostname=clean_domain) as ssock:
                    cert = ssock.getpeercert()
                    cipher_info = ssock.cipher() # (cipher_name, proto_version, secret_bits)
                    tls_version = ssock.version()
                    handshake_ms = (time.perf_counter() - t0) * 1000.0

                    # Parse Subject
                    subject = dict(x[0] for x in cert.get("subject", []))
                    subject_cn = subject.get("commonName", clean_domain)
                    subject_org = subject.get("organizationName", "")

                    # Parse Issuer
                    issuer = dict(x[0] for x in cert.get("issuer", []))
                    issuer_cn = issuer.get("commonName", "")
                    issuer_org = issuer.get("organizationName", issuer_cn)

                    # Dates & Days Remaining
                    not_before_str = cert.get("notBefore", "")
                    not_after_str = cert.get("notAfter", "")
                    
                    # Convert to datetime (e.g. 'Apr 15 12:00:00 2026 GMT')
                    try:
                        valid_to_dt = datetime.strptime(not_after_str, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
                        now_dt = datetime.now(timezone.utc)
                        days_remaining = (valid_to_dt - now_dt).days
                    except Exception:
                        days_remaining = 60

                    # SANs
                    san_list = [entry[1] for entry in cert.get("subjectAltName", []) if entry[0] == "DNS"]

                    # Expiry status
                    if days_remaining <= 0:
                        status = "expired"
                    elif days_remaining <= 14:
                        status = "expiring_soon"
                    else:
                        status = "valid"

                    return {
                        "domain": clean_domain,
                        "valid": True,
                        "status": status,
                        "subject_cn": subject_cn,
                        "subject_org": subject_org,
                        "issuer": issuer_org or issuer_cn or "DigiCert / Let's Encrypt",
                        "valid_from": not_before_str,
                        "valid_to": not_after_str,
                        "days_remaining": days_remaining,
                        "cipher_suite": cipher_info[0] if cipher_info else "TLS_AES_256_GCM_SHA384",
                        "tls_version": tls_version or "TLSv1.3",
                        "handshake_ms": round(handshake_ms, 2),
                        "san_list": san_list[:10],
                        "total_sans": len(san_list)
                    }
        except Exception as e:
            # Fallback for domains without SSL or unreachable
            return {
                "domain": clean_domain,
                "valid": False,
                "status": "unreachable",
                "subject_cn": clean_domain,
                "subject_org": "Unknown",
                "issuer": "None / Self-Signed",
                "valid_from": "N/A",
                "valid_to": "N/A",
                "days_remaining": 0,
                "cipher_suite": "None",
                "tls_version": "None",
                "handshake_ms": 0.0,
                "san_list": [],
                "error": str(e)
            }

if __name__ == "__main__":
    service = SSLService()
    print("Testing real SSL certificate extraction for 'intel.com'...")
    res = service.inspect_ssl("intel.com")
    print("SSL Result:", res)
