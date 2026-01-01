#!/usr/bin/env python3
"""
Comprehensive Test Suite for EEG Visualization System
Tests all 8 visualization types with real CSV data
"""

import os
import sys
import django
import requests
import time
from pathlib import Path

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NeuroPi.settings')
django.setup()

from django.test import TestCase, Client
from django.urls import reverse
from eeg_visualization.views import *

class EEGVisualizationTestSuite:
    """Complete test suite for EEG visualization system"""
    
    def __init__(self):
        self.client = Client()
        self.base_url = "http://127.0.0.1:8000"
        self.test_results = {}
        
    def test_home_page(self):
        """Test main EEG visualization page (page 333)"""
        print("🧪 Testing Home Page...")
        try:
            response = self.client.get('/eeg-visualization/')
            assert response.status_code == 200
            assert 'Advanced Brain Signal Analysis' in response.content.decode()
            print("✅ Home page loads successfully")
            return True
        except Exception as e:
            print(f"❌ Home page test failed: {e}")
            return False
    
    def test_csv_upload(self):
        """Test CSV file upload functionality"""
        print("🧪 Testing CSV Upload...")
        try:
            # Test default data loading
            response = self.client.get('/eeg-visualization/load-default/')
            assert response.status_code == 200
            print("✅ Default CSV data loaded successfully")
            return True
        except Exception as e:
            print(f"❌ CSV upload test failed: {e}")
            return False
    
    def test_visualization_plots(self):
        """Test all 8 visualization plot types"""
        plot_types = [
            ('interactive', 'Interactive Plots'),
            ('topographic', 'Topographic Maps'),
            ('timefreq', 'Time-Frequency Analysis'),
            ('3d', '3D Visualization'),
            ('ica', 'ICA Components'),
            ('stats', 'Statistical Plots'),
            ('connectivity', 'Brain Connectivity'),
            ('psd', 'Power Spectral Density')
        ]
        
        results = {}
        
        for plot_type, plot_name in plot_types:
            print(f"🧪 Testing {plot_name}...")
            try:
                response = self.client.get(f'/eeg-visualization/plot/{plot_type}/')
                
                if response.status_code == 200:
                    content = response.content.decode()
                    
                    # Check if English interface is working
                    if plot_name in content:
                        print(f"✅ {plot_name} loads successfully with English interface")
                        results[plot_type] = "PASS"
                    else:
                        print(f"⚠️ {plot_name} loads but interface may not be fully English")
                        results[plot_type] = "PARTIAL"
                else:
                    print(f"❌ {plot_name} failed to load (Status: {response.status_code})")
                    results[plot_type] = "FAIL"
                    
            except Exception as e:
                print(f"❌ {plot_name} test failed: {e}")
                results[plot_type] = "ERROR"
        
        return results
    
    def test_plot_generation(self):
        """Test actual plot generation with CSV data"""
        print("🧪 Testing Plot Generation with Real Data...")
        
        # First load default data
        self.client.get('/eeg-visualization/load-default/')
        
        # Test plots that should generate actual images
        test_plots = ['interactive', 'topographic', 'timefreq', 'psd']
        results = {}
        
        for plot_type in test_plots:
            try:
                response = self.client.get(f'/eeg-visualization/plot/{plot_type}/')
                content = response.content.decode()
                
                # Check if plot image is generated
                if 'data:image/png;base64,' in content:
                    print(f"✅ {plot_type} generates actual plot image")
                    results[plot_type] = "PLOT_GENERATED"
                elif 'Upload CSV data to generate visualization' in content:
                    print(f"⚠️ {plot_type} shows placeholder (no data)")
                    results[plot_type] = "NO_DATA"
                else:
                    print(f"❌ {plot_type} no plot generated")
                    results[plot_type] = "NO_PLOT"
                    
            except Exception as e:
                print(f"❌ {plot_type} plot generation failed: {e}")
                results[plot_type] = "ERROR"
        
        return results
    
    def test_csv_file_structure(self):
        """Test CSV file structure and data integrity"""
        print("🧪 Testing CSV File Structure...")
        
        csv_path = "Trials_data/visual_trial_test/session_25/visual_trial_data_20250704_204814.csv"
        
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                # Check required columns
                required_columns = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 
                                  'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'Timestamp']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if not missing_columns:
                    print(f"✅ CSV file structure is correct ({len(df)} rows, {len(df.columns)} columns)")
                    print(f"📊 Data range: {len(df)} samples")
                    return True
                else:
                    print(f"❌ Missing columns: {missing_columns}")
                    return False
            else:
                print(f"❌ CSV file not found: {csv_path}")
                return False
                
        except Exception as e:
            print(f"❌ CSV structure test failed: {e}")
            return False
    
    def run_complete_test(self):
        """Run complete test suite"""
        print("🚀 Starting EEG Visualization System Test Suite")
        print("=" * 60)
        
        # Test results storage
        all_results = {}
        
        # 1. Test home page
        all_results['home_page'] = self.test_home_page()
        print()
        
        # 2. Test CSV functionality
        all_results['csv_upload'] = self.test_csv_upload()
        print()
        
        # 3. Test CSV file structure
        all_results['csv_structure'] = self.test_csv_file_structure()
        print()
        
        # 4. Test all visualization pages
        all_results['visualization_plots'] = self.test_visualization_plots()
        print()
        
        # 5. Test plot generation
        all_results['plot_generation'] = self.test_plot_generation()
        print()
        
        # Generate summary report
        self.generate_test_report(all_results)
        
        return all_results
    
    def generate_test_report(self, results):
        """Generate comprehensive test report"""
        print("📋 TEST SUMMARY REPORT")
        print("=" * 60)
        
        # Count results
        total_tests = 0
        passed_tests = 0
        
        # Home page test
        if results['home_page']:
            print("✅ Home Page: PASS")
            passed_tests += 1
        else:
            print("❌ Home Page: FAIL")
        total_tests += 1
        
        # CSV tests
        if results['csv_upload']:
            print("✅ CSV Upload: PASS")
            passed_tests += 1
        else:
            print("❌ CSV Upload: FAIL")
        total_tests += 1
        
        if results['csv_structure']:
            print("✅ CSV Structure: PASS")
            passed_tests += 1
        else:
            print("❌ CSV Structure: FAIL")
        total_tests += 1
        
        # Visualization plot tests
        plot_results = results['visualization_plots']
        for plot_type, result in plot_results.items():
            if result == "PASS":
                print(f"✅ {plot_type.title()} Plot: PASS")
                passed_tests += 1
            elif result == "PARTIAL":
                print(f"⚠️ {plot_type.title()} Plot: PARTIAL")
                passed_tests += 0.5
            else:
                print(f"❌ {plot_type.title()} Plot: {result}")
            total_tests += 1
        
        # Plot generation tests
        gen_results = results['plot_generation']
        for plot_type, result in gen_results.items():
            if result == "PLOT_GENERATED":
                print(f"✅ {plot_type.title()} Image: GENERATED")
                passed_tests += 1
            else:
                print(f"❌ {plot_type.title()} Image: {result}")
            total_tests += 1
        
        print("=" * 60)
        success_rate = (passed_tests / total_tests) * 100
        print(f"📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🎉 EXCELLENT: System is working very well!")
        elif success_rate >= 75:
            print("👍 GOOD: System is working well with minor issues")
        elif success_rate >= 50:
            print("⚠️ FAIR: System has some issues that need attention")
        else:
            print("❌ POOR: System needs significant fixes")

if __name__ == "__main__":
    # Run the test suite
    tester = EEGVisualizationTestSuite()
    results = tester.run_complete_test()
