import subprocess
import time
import json
import urllib.request
import urllib.error
import sys
def run_tests():
    print("Starting Uvicorn server subprocess...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Poll /health until server is responsive
    print("Waiting for server to become responsive...")
    base_url = "http://127.0.0.1:8000"
    started = False
    for i in range(20):
        if server_process.poll() is not None:
            print("Error: Server process exited early.")
            break
        try:
            req = urllib.request.Request(f"{base_url}/health")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    started = True
                    print(f"Server is up and running (responsive after {i*0.5:.1f}s)!")
                    break
        except Exception:
            time.sleep(0.5)
            
    if not started:
        print("Error: Uvicorn server failed to start or become responsive. Logs:")
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
        return False

    success = True
    try:
        # Test 1: GET /health
        print("\n--- Test 1: GET /health ---")
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("Response status code:", resp.status)
            print("Response payload:", json.dumps(data, indent=2))
            assert resp.status == 200
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            print("SUCCESS")
            
        # Test 2: GET /model/info
        print("\n--- Test 2: GET /model/info ---")
        req = urllib.request.Request(f"{base_url}/model/info")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("Response status code:", resp.status)
            print("Response keys:", list(data.keys()))
            assert resp.status == 200
            assert "model_version" in data
            assert "training_data_hash" in data
            print("SUCCESS")
            
        # Test 3: POST /score (Valid transaction)
        print("\n--- Test 3: POST /score (Valid) ---")
        valid_payload = {
            "Time": 1000.0, "Amount": 500.0,
            "V1": -1.0, "V2": 0.5, "V3": 1.0, "V4": 0.8, "V5": -0.2, "V6": -0.4, "V7": 0.5, "V8": 0.1,
            "V9": -0.3, "V10": -0.8, "V11": 0.9, "V12": -1.2, "V13": 0.2, "V14": -2.0, "V15": 0.1, "V16": -0.9,
            "V17": -1.5, "V18": -0.6, "V19": 0.2, "V20": 0.05, "V21": 0.1, "V22": -0.05, "V23": -0.1, "V24": 0.05,
            "V25": 0.2, "V26": -0.05, "V27": 0.1, "V28": -0.02
        }
        req = urllib.request.Request(
            f"{base_url}/score",
            data=json.dumps(valid_payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("Response status code:", resp.status)
            print("Response payload:", json.dumps(data, indent=2))
            assert resp.status == 200
            assert "fraud_probability" in data
            assert "risk_level" in data
            print("SUCCESS")

        # Test 4: POST /score (Invalid inputs - expects 422 validation error)
        print("\n--- Test 4: POST /score (Invalid Input: negative Amount) ---")
        invalid_payload = valid_payload.copy()
        invalid_payload["Amount"] = -5.0
        req = urllib.request.Request(
            f"{base_url}/score",
            data=json.dumps(invalid_payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as resp:
                print("FAILED: Server accepted negative amount!")
                success = False
        except urllib.error.HTTPError as e:
            print("Response status code (expected 422):", e.code)
            body = json.loads(e.read().decode())
            print("Validation errors:", json.dumps(body, indent=2))
            assert e.code == 422
            print("SUCCESS")

        # Test 5: POST /explain (Flagged transaction explanation)
        print("\n--- Test 5: POST /explain (Valid explanation request) ---")
        flagged_payload = valid_payload.copy()
        flagged_payload["V14"] = -8.5
        flagged_payload["V10"] = -6.0
        flagged_payload["V12"] = -5.0
        
        req = urllib.request.Request(
            f"{base_url}/explain",
            data=json.dumps(flagged_payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("Response status code:", resp.status)
            print("Response probability:", data["fraud_probability"])
            print("Response explanation layer output:")
            print(json.dumps(data["explanation"], indent=2))
            assert resp.status == 200
            assert "fraud_probability" in data
            assert "explanation" in data
            assert "risk_level" in data["explanation"]
            print("SUCCESS")

    except Exception as e:
        print("\nFAILED: An unexpected error occurred during test runs:", e)
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            print("=== SERVER STDOUT LOGS ===")
            print(stdout)
            print("=== SERVER STDERR LOGS ===")
            print(stderr)
        success = False
    finally:
        print("\nTerminating Uvicorn server subprocess...")
        if server_process.poll() is None:
            server_process.terminate()
            stdout, stderr = server_process.communicate()
        print("Server shutdown completed.")
        
    return success

if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
