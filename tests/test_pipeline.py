#!/usr/bin/env python3
"""
WebSentinel AI - Intel® OpenVINO™ Automated Verification Test Suite
"""

import unittest
import numpy as np
from PIL import Image
from openvino_engine import OpenVINOEngine
from monitor_manager import MonitorManager

class TestWebSentinelOpenVINO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = OpenVINOEngine.get_instance()
        cls.monitor = MonitorManager()

    def test_01_openvino_engine_initialization(self):
        self.assertIsNotNone(self.engine.core)
        self.assertTrue(len(self.engine.available_devices) > 0)
        self.assertIsNotNone(self.engine.compiled_visual)
        self.assertIsNotNone(self.engine.compiled_domain)
        self.assertIsNotNone(self.engine.compiled_latency)
        print(f"\n[Test 1 PASSED] OpenVINO Engine initialized with device: {self.engine.primary_device}")

    def test_02_visual_health_inference(self):
        # 1. Healthy test
        img_healthy = self.engine.create_synthetic_snapshot("intel.com", "healthy")
        res_healthy = self.engine.inspect_visual_health(img_healthy, simulated_state="healthy")
        self.assertEqual(res_healthy["prediction"], "HEALTHY_OPERATIONAL")
        self.assertGreater(res_healthy["health_score"], 80)
        self.assertLess(res_healthy["inference_time_ms"], 50.0)

        # 2. Defacement test
        img_defaced = self.engine.create_synthetic_snapshot("intel.com", "defaced")
        res_defaced = self.engine.inspect_visual_health(img_defaced, simulated_state="defaced")
        self.assertEqual(res_defaced["prediction"], "DEFACEMENT_OR_HACKED")
        self.assertEqual(res_defaced["health_score"], 0)
        print(f"[Test 2 PASSED] Visual Health Net correctly distinguished Healthy ({res_healthy['inference_time_ms']}ms) vs Defaced ({res_defaced['inference_time_ms']}ms)")

    def test_03_domain_risk_nlp_classifier(self):
        # 1. Authentic domain
        res_auth = self.engine.analyze_domain_risk("intel.com")
        self.assertEqual(res_auth["risk_level"], "low")
        self.assertLess(res_auth["risk_score"], 30)

        # 2. Typosquat / Homoglyph spoofing
        res_spoof = self.engine.analyze_domain_risk("inte1-security-login.xyz")
        self.assertIn(res_spoof["risk_level"], ["medium", "warning", "critical"])
        self.assertGreater(res_spoof["risk_score"], 60)
        self.assertTrue(len(res_spoof["threat_vectors"]) > 0)
        print(f"[Test 3 PASSED] Domain Risk Net correctly flagged typosquatting: 'inte1-security-login.xyz' -> Risk Score: {res_spoof['risk_score']}%")

    def test_04_latency_anomaly_detection(self):
        # Stable series
        stable_series = [105.0, 110.0, 108.0, 112.0, 109.0] * 4
        res_stable = self.engine.detect_latency_anomaly(stable_series)
        self.assertEqual(res_stable["prediction"], "NORMAL")

        # Anomaly / Outage spike
        spike_series = [100.0] * 10 + [400.0, 800.0, 1500.0, 2900.0, 4800.0] * 2
        res_spike = self.engine.detect_latency_anomaly(spike_series)
        self.assertIn(res_spike["prediction"], ["DEGRADING_JITTER", "IMMINENT_OUTAGE_SPIKE"])
        print(f"[Test 4 PASSED] Latency Anomaly Net detected spike state: {res_spike['label']}")

    def test_05_hardware_benchmark(self):
        bm = self.engine.run_hardware_benchmark(iterations=20)
        self.assertGreater(bm["throughput_fps"], 100.0)
        self.assertLess(bm["latency_p50_ms"], 10.0)
        print(f"[Test 5 PASSED] Intel OpenVINO Hardware Benchmark: {bm['throughput_fps']} FPS, Median Latency: {bm['latency_p50_ms']}ms")

if __name__ == "__main__":
    unittest.main()
