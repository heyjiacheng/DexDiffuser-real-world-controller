"""
Simple test script to verify the API service is working
"""
import requests
import sys

def test_api_server(server_url="http://100.120.117.28:8000"):
    """Test basic API endpoints"""

    print(f"Testing API server at {server_url}")
    print("="*50)

    # Test 1: Root endpoint
    try:
        print("\n1. Testing root endpoint...")
        response = requests.get(f"{server_url}/", timeout=5)
        if response.status_code == 200:
            print("   ✓ Root endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ✗ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   Make sure the server is running!")
        return False

    # Test 2: Health check
    try:
        print("\n2. Testing health check...")
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("   ✓ Health check passed")
            print(f"   Status: {health_data['status']}")
            print(f"   Model loaded: {health_data['model_loaded']}")
            if not health_data['model_loaded']:
                print("   ⚠ Warning: DexDiffuser model not loaded!")
        else:
            print(f"   ✗ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 3: Sessions list
    try:
        print("\n3. Testing sessions endpoint...")
        response = requests.get(f"{server_url}/sessions", timeout=5)
        if response.status_code == 200:
            sessions_data = response.json()
            print("   ✓ Sessions endpoint working")
            print(f"   Active sessions: {sessions_data['active_sessions']}")
        else:
            print(f"   ✗ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print("\n" + "="*50)
    print("All basic tests passed! ✓")
    print("\nServer is ready to receive grasp generation requests.")
    print("Use client_example.py to send RGB-D data and generate grasps.")
    return True

if __name__ == "__main__":
    # Get server URL from command line or use default
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://100.120.117.28:8000"

    success = test_api_server(server_url)
    sys.exit(0 if success else 1)
