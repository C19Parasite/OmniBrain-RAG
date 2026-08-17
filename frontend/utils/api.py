import requests

BASE_URL = "http://127.0.0.1:8000"


def upload_file(uploaded_file):
    try:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            },
        )

        if response.status_code == 200:
            data = response.json()

            return {
                "success": True,
                "message": data.get("message", "File uploaded successfully"),
            }

        return {
            "success": False,
            "message": f"Upload failed: {response.status_code}",
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "message": f"Could not connect to backend: {e}",
        }

        
def query_api(question):
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={
                "query": question
            },
        )

        if response.status_code == 200:
            data = response.json()

            return {
                "success": True,
                "answer": data.get("context", "No context received"),
            }

        return {
            "success": False,
            "answer": f"Query failed: {response.status_code}",
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "answer": f"Could not connect to backend: {e}",
        }