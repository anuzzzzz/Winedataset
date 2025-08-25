import time
import requests
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

class WineAPILoadTester:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.results = []
        self.test_data = {
            "alcohol": 13.0,
            "malic_acid": 2.0,
            "ash": 2.5,
            "alcalinity_of_ash": 20.0,
            "magnesium": 100.0,
            "total_phenols": 2.5,
            "flavanoids": 2.0,
            "nonflavanoid_phenols": 0.3,
            "proanthocyanins": 1.5,
            "color_intensity": 5.0,
            "hue": 1.0,
            "od280/od315_of_diluted_wines": 2.5,
            "proline": 800.0,
            "location": 0
        }
    
    def make_request(self, endpoint="/predict", data=None):
        """Make a single API request and record metrics"""
        start_time = time.time()
        
        try:
            if endpoint == "/health":
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
            else:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=data or self.test_data,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
            
            response_time = time.time() - start_time
            
            result = {
                "timestamp": time.time(),
                "endpoint": endpoint,
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200
            }
            
            if response.status_code == 200 and endpoint == "/predict":
                try:
                    result["prediction"] = response.json().get("prediction", -1)
                    result["probability"] = response.json().get("probability", 0)
                except:
                    result["prediction"] = -1
                    result["probability"] = 0
            
            return result
            
        except Exception as e:
            return {
                "timestamp": time.time(),
                "endpoint": endpoint,
                "status_code": 0,
                "response_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
    
    def worker_thread(self, duration, requests_per_second):
        """Worker thread for load testing"""
        end_time = time.time() + duration
        thread_results = []
        
        while time.time() < end_time:
            # Make request
            result = self.make_request()
            thread_results.append(result)
            
            # Wait to achieve target RPS
            time.sleep(1.0 / requests_per_second)
        
        return thread_results
    
    def run_load_test(self, duration=60, concurrent_users=5, requests_per_second=2):
        """Run comprehensive load test"""
        print(f"🚀 STARTING LOAD TEST")
        print(f"   Duration: {duration} seconds")
        print(f"   Concurrent users: {concurrent_users}")
        print(f"   Requests per second per user: {requests_per_second}")
        print(f"   Total expected requests: ~{duration * concurrent_users * requests_per_second}")
        
        # Start load test
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            for i in range(concurrent_users):
                future = executor.submit(self.worker_thread, duration, requests_per_second)
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in futures:
                thread_results = future.result()
                all_results.extend(thread_results)
        
        total_time = time.time() - start_time
        self.results = all_results
        
        # Analyze results
        self.analyze_results(total_time)
        self.create_visualizations()
        
        return self.results
    
    def analyze_results(self, total_time):
        """Analyze load test results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        
        # Basic statistics
        total_requests = len(df)
        successful_requests = len(df[df['success']])
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests) * 100
        
        # Response time statistics
        successful_df = df[df['success']]
        if len(successful_df) > 0:
            avg_response_time = successful_df['response_time'].mean()
            p50_response_time = successful_df['response_time'].quantile(0.5)
            p95_response_time = successful_df['response_time'].quantile(0.95)
            p99_response_time = successful_df['response_time'].quantile(0.99)
            max_response_time = successful_df['response_time'].max()
            min_response_time = successful_df['response_time'].min()
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0
            max_response_time = min_response_time = 0
        
        # Throughput
        actual_rps = total_requests / total_time
        successful_rps = successful_requests / total_time
        
        # Status code distribution
        status_codes = df['status_code'].value_counts().to_dict()
        
        print(f"\n📊 LOAD TEST RESULTS")
        print("=" * 50)
        print(f"📈 PERFORMANCE METRICS:")
        print(f"   Total Requests: {total_requests}")
        print(f"   Successful: {successful_requests} ({success_rate:.1f}%)")
        print(f"   Failed: {failed_requests}")
        print(f"   Actual RPS: {actual_rps:.2f}")
        print(f"   Successful RPS: {successful_rps:.2f}")
        
        print(f"\n⏱️  RESPONSE TIME METRICS:")
        print(f"   Average: {avg_response_time:.3f}s")
        print(f"   Median (P50): {p50_response_time:.3f}s")
        print(f"   P95: {p95_response_time:.3f}s")
        print(f"   P99: {p99_response_time:.3f}s")
        print(f"   Min: {min_response_time:.3f}s")
        print(f"   Max: {max_response_time:.3f}s")
        
        print(f"\n🔍 STATUS CODE DISTRIBUTION:")
        for code, count in sorted(status_codes.items()):
            print(f"   {code}: {count} ({count/total_requests*100:.1f}%)")
        
        # Prediction accuracy check
        if 'prediction' in df.columns:
            predictions = df[df['prediction'] >= 0]['prediction'].value_counts().to_dict()
            if predictions:
                print(f"\n🎯 PREDICTION DISTRIBUTION:")
                for pred, count in sorted(predictions.items()):
                    cultivar_name = f"Cultivar {pred}"
                    print(f"   {cultivar_name}: {count}")
        
        # Save results
        results_summary = {
            "total_requests": int(total_requests),
            "successful_requests": int(successful_requests),
            "failed_requests": int(failed_requests),
            "success_rate": float(success_rate),
            "actual_rps": float(actual_rps),
            "successful_rps": float(successful_rps),
            "response_times": {
                "average": float(avg_response_time),
                "median": float(p50_response_time),
                "p95": float(p95_response_time),
                "p99": float(p99_response_time),
                "min": float(min_response_time),
                "max": float(max_response_time)
            },
            "status_codes": status_codes
        }
        
        with open("reports/load_test_results.json", "w") as f:
            json.dump(results_summary, f, indent=2)
        
        # Save detailed results
        df.to_csv("reports/load_test_detailed.csv", index=False)
        
        print(f"\n✅ Results saved to reports/load_test_results.json")
    
    def create_visualizations(self):
        """Create load test visualization plots"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Response time over time
        ax1 = axes[0, 0]
        successful_df = df[df['success']].copy()
        if len(successful_df) > 0:
            # Convert timestamp to elapsed time
            start_time = df['timestamp'].min()
            successful_df['elapsed'] = successful_df['timestamp'] - start_time
            
            # Plot response time
            ax1.scatter(successful_df['elapsed'], successful_df['response_time'], 
                       alpha=0.6, s=20)
            ax1.set_xlabel('Elapsed Time (seconds)')
            ax1.set_ylabel('Response Time (seconds)')
            ax1.set_title('Response Time Over Test Duration')
            ax1.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(successful_df['elapsed'], successful_df['response_time'], 1)
            p = np.poly1d(z)
            ax1.plot(successful_df['elapsed'], p(successful_df['elapsed']), 
                    "r--", alpha=0.8, linewidth=2, label=f'Trend')
            ax1.legend()
        
        # 2. Response time histogram
        ax2 = axes[0, 1]
        if len(successful_df) > 0:
            ax2.hist(successful_df['response_time'], bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(successful_df['response_time'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {successful_df["response_time"].mean():.3f}s')
            ax2.axvline(successful_df['response_time'].quantile(0.95), color='orange', 
                       linestyle='--', linewidth=2, label=f'P95: {successful_df["response_time"].quantile(0.95):.3f}s')
            ax2.set_xlabel('Response Time (seconds)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Response Time Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Success rate over time
        ax3 = axes[1, 0]
        df_windowed = df.copy()
        df_windowed['elapsed'] = df_windowed['timestamp'] - df_windowed['timestamp'].min()
        
        # Calculate rolling success rate
        window_size = max(10, len(df) // 20)  # Adaptive window size
        df_windowed = df_windowed.sort_values('elapsed')
        df_windowed['success_rate_rolling'] = df_windowed['success'].rolling(
            window=window_size, min_periods=1).mean() * 100
        
        ax3.plot(df_windowed['elapsed'], df_windowed['success_rate_rolling'], 
                linewidth=2, color='green')
        ax3.set_xlabel('Elapsed Time (seconds)')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title(f'Rolling Success Rate (Window: {window_size} requests)')
        ax3.set_ylim([0, 105])
        ax3.grid(True, alpha=0.3)
        
        # 4. Throughput over time
        ax4 = axes[1, 1]
        # Calculate throughput in time windows
        time_windows = np.arange(0, df_windowed['elapsed'].max(), 10)  # 10-second windows
        throughput_data = []
        
        for i in range(len(time_windows) - 1):
            window_start = time_windows[i]
            window_end = time_windows[i + 1]
            
            window_requests = df_windowed[
                (df_windowed['elapsed'] >= window_start) & 
                (df_windowed['elapsed'] < window_end)
            ]
            
            throughput = len(window_requests) / 10  # RPS in this window
            throughput_data.append({
                'time': window_start + 5,  # Middle of window
                'rps': throughput
            })
        
        if throughput_data:
            throughput_df = pd.DataFrame(throughput_data)
            ax4.plot(throughput_df['time'], throughput_df['rps'], 
                    linewidth=2, marker='o', markersize=4)
            ax4.set_xlabel('Elapsed Time (seconds)')
            ax4.set_ylabel('Requests per Second')
            ax4.set_title('Throughput Over Time (10s windows)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/load_test_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Load test visualizations saved to reports/load_test_analysis.png")

def run_comprehensive_load_test():
    """Run the comprehensive load testing suite"""
    print("🧪 COMPREHENSIVE LOAD TESTING SUITE")
    print("=" * 50)
    
    # Check if API is accessible
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code != 200:
            print("❌ API health check failed. Make sure the API is running.")
            return
    except:
        print("❌ Cannot connect to API. Starting port-forward...")
        import subprocess
        # Start port-forward
        proc = subprocess.Popen([
            'kubectl', 'port-forward', 
            'service/wine-classifier-service', 
            '8080:80', '-n', 'wine-classifier'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(5)
        
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            if response.status_code != 200:
                print("❌ API still not accessible")
                proc.terminate()
                return
        except:
            print("❌ API connection failed")
            proc.terminate()
            return
    
    print("✅ API is accessible, starting load tests...")
    
    # Initialize tester
    tester = WineAPILoadTester()
    
    # Run different load test scenarios
    test_scenarios = [
        {"name": "Light Load", "duration": 30, "users": 3, "rps": 2},
        {"name": "Medium Load", "duration": 60, "users": 5, "rps": 3},
        {"name": "Heavy Load", "duration": 30, "users": 8, "rps": 4}
    ]
    
    all_results = {}
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎯 SCENARIO {i}: {scenario['name']}")
        print("-" * 30)
        
        results = tester.run_load_test(
            duration=scenario['duration'],
            concurrent_users=scenario['users'],
            requests_per_second=scenario['rps']
        )
        
        all_results[scenario['name']] = results
        
        if i < len(test_scenarios):
            print("\n⏸️  Cooling down for 10 seconds...")
            time.sleep(10)
    
    # Generate final summary
    print(f"\n🎉 LOAD TESTING COMPLETED!")
    print("=" * 50)
    
    return all_results

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    run_comprehensive_load_test()
