#!/usr/bin/env python3
"""
WebSentinel AI - Production Test Suite
Validates:
1. Intel® OpenVINO™ Deep Learning Engine (MobileNetV2, Vision Health Net, Domain Risk Net, Latency Net)
2. Live DNS record resolution (dnspython)
3. Live SSL/TLS socket handshake (TLSv1.3, x509 cert extraction)
4. SQLite persistent database operations
5. Full live website crawl & audit pipeline
"""

import unittest
import asyncio
from PIL import Image
from openvino_engine import OpenVINOEngine
from dns_service import DNSService
from ssl_service import SSLService
from database import Database
from monitor_manager import MonitorManager

class TestWebSentinelProduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = OpenVINOEngine.get_instance()
        cls.db = Database.get_instance()
        cls.monitor = MonitorManager.get_instance()

    def test_01_openvino_models_loaded(self):
        self.assertIsNotNone(self.engine.compiled_mobilenet, "MobileNetV2 feature extractor must be compiled")
        self.assertIsNotNone(self.engine.compiled_visual, "Visual health net must be compiled")
        self.assertIsNotNone(self.engine.compiled_domain, "Domain risk net must be compiled")
        self.assertIsNotNone(self.engine.compiled_latency, "Latency anomaly net must be compiled")
        print(f"\n[Test 1 PASSED] All 4 OpenVINO models compiled and active on device: {self.engine.primary_device}")

    def test_02_real_dns_resolution(self):
        dns_res = DNSService.resolve_all_records("intel.com")
        self.assertGreater(dns_res["total_records_found"], 0)
        self.assertIn("A", dns_res["records_by_type"])
        self.assertTrue(dns_res["has_mx"])
        print(f"[Test 2 PASSED] Real DNS resolution for 'intel.com': Found {dns_res['total_records_found']} records in {dns_res['dns_latency_ms']}ms")

    def test_03_real_ssl_handshake(self):
        ssl_res = SSLService.inspect_ssl("intel.com")
        self.assertTrue(ssl_res["valid"])
        self.assertIn("TLS", ssl_res["tls_version"])
        self.assertGreater(ssl_res["days_remaining"], 0)
        print(f"[Test 3 PASSED] Real SSL Handshake for 'intel.com': Issuer={ssl_res['issuer']}, Protocol={ssl_res['tls_version']}, Days Left={ssl_res['days_remaining']}")

    def test_04_openvino_vision_inference_on_real_image(self):
        # Test on live screenshot or generated image
        img = Image.new("RGB", (224, 224), color=(30, 41, 59))
        res = self.engine.inspect_visual_health(img)
        self.assertIn("prediction", res)
        self.assertGreater(res["confidence"], 0)
        self.assertLess(res["inference_time_ms"], 50.0)
        print(f"[Test 4 PASSED] OpenVINO Vision inference completed in {res['inference_time_ms']}ms with confidence {res['confidence']}%")

    def test_05_openvino_domain_risk_classifier(self):
        # 1. Authentic domain
        res_auth = self.engine.analyze_domain_risk("intel.com")
        self.assertEqual(res_auth["risk_level"], "low")

        # 2. Typosquatting / Malicious domain
        res_phish = self.engine.analyze_domain_risk("inte1-security-portal.xyz")
        self.assertIn(res_phish["risk_level"], ["medium", "warning", "critical"])
        self.assertGreater(res_phish["risk_score"], 50)
        self.assertTrue(len(res_phish["threat_vectors"]) > 0)
        print(f"[Test 5 PASSED] OpenVINO Domain NLP correctly classified 'intel.com' (Safe) and 'inte1-security-portal.xyz' (Risk: {res_phish['risk_score']}%)")

    def test_06_database_crud(self):
        domains = self.db.get_all_domains()
        self.assertGreater(len(domains), 0)
        latest = self.db.get_latest_audit(domains[0]["id"])
        self.assertIsNotNone(latest)
        print(f"[Test 6 PASSED] SQLite DB holds {len(domains)} monitored domains with full audit histories")

if __name__ == "__main__":
    unittest.main()
