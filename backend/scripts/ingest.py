#!/usr/bin/env python3
"""
Helper script to trigger and monitor ingestion jobs via the API.
Usage: python scripts/ingest.py [http://localhost:8000]
"""

import sys
import time
import requests
import json

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    base_url = base_url.rstrip("/")
    
    print(f"🔌 Connecting to {base_url}...")

    # 1. Health Check
    try:
        resp = requests.get(f"{base_url}/health")
        resp.raise_for_status()
        print("✅ Backend is up and running.")
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {base_url}. Is the backend running?")
        print("   Try: uvicorn main:app --reload")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)

    # 2. Trigger Ingestion
    print("\n🚀 Triggering ingestion job...")
    try:
        resp = requests.post(f"{base_url}/ingest")
        resp.raise_for_status()
        job_data = resp.json()
        job_id = job_data["job_id"]
        print(f"✅ Job started! ID: {job_id}")
    except Exception as e:
        print(f"❌ Failed to start ingestion: {e}")
        sys.exit(1)

    # 3. Poll Status
    print("\n⏳ Polling for completion...")
    start_time = time.time()
    
    while True:
        try:
            resp = requests.get(f"{base_url}/ingest/{job_id}")
            resp.raise_for_status()
            job = resp.json()
            status = job["status"]
            
            elapsed = int(time.time() - start_time)
            print(f"   [{elapsed}s] Status: {status.upper()}")
            
            if status == "completed":
                print(f"\n✅ Ingestion completed! Data is now live.")
                break
            elif status == "failed":
                print(f"\n❌ Ingestion failed!")
                if job.get("error"):
                    print(f"   Error: {job['error']}")
                sys.exit(1)
            elif status == "cancelled":
                print(f"\n⚠️ Ingestion cancelled.")
                sys.exit(1)
                
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped (Job continues in background).")
            break
        except Exception as e:
            print(f"❌ Error polling status: {e}")
            break

if __name__ == "__main__":
    main()
