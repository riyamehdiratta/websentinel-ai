#!/usr/bin/env python3
"""
WebSentinel AI - Real DNS Resolution & Diagnostics Service
Performs live DNS queries across authoritative nameservers for A, AAAA, MX, NS, TXT, CNAME, SOA.
"""

import time
import dns.resolver
from typing import Dict, Any, List

RECORD_TYPES = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME', 'SOA']

class DNSService:
    @staticmethod
    def resolve_all_records(domain: str) -> Dict[str, Any]:
        """
        Queries live DNS records for any real internet domain.
        """
        clean_domain = domain.lower().strip().replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
        
        resolver = dns.resolver.Resolver()
        resolver.nameservers = ['8.8.8.8', '1.1.1.1', '8.8.4.4']
        resolver.timeout = 2.0
        resolver.lifetime = 4.0

        records_by_type = {}
        all_records_list = []
        
        t0 = time.perf_counter()
        
        for rtype in RECORD_TYPES:
            try:
                answers = resolver.resolve(clean_domain, rtype)
                entries = []
                for rdata in answers:
                    val = str(rdata).strip('"')
                    ttl = answers.rrset.ttl if answers.rrset else 300
                    entries.append({"value": val, "ttl": ttl})
                    all_records_list.append({"type": rtype, "value": val, "ttl": ttl})
                if entries:
                    records_by_type[rtype] = entries
            except Exception:
                # Type not configured or NXDOMAIN
                pass

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "domain": clean_domain,
            "dns_latency_ms": round(elapsed_ms, 2),
            "records_by_type": records_by_type,
            "all_records": all_records_list,
            "total_records_found": len(all_records_list),
            "has_mx": 'MX' in records_by_type,
            "has_txt_spf": any('v=spf1' in r['value'] for r in records_by_type.get('TXT', []))
        }

if __name__ == "__main__":
    service = DNSService()
    print("Testing real DNS resolution for 'intel.com'...")
    res = service.resolve_all_records("intel.com")
    print(f"Resolved {res['total_records_found']} records in {res['dns_latency_ms']}ms:")
    for rtype, records in res["records_by_type"].items():
        print(f"[{rtype}]: {[r['value'] for r in records]}")
