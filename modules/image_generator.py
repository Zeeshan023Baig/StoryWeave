import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

_PIXAZO_URL = "https://gateway.pixazo.ai/flux/text-to-image"
_STATUS_URL_TEMPLATE = "https://gateway.pixazo.ai/v2/requests/status/{request_id}"
_MAX_RETRIES = 10
_POLL_INTERVAL_SEC = 2

def generate_image_bytes(prompt):
    """
    Generates an educational comic illustration from a text prompt using the
    Pixazo Flux API and returns raw PNG/JPEG bytes for st.image().

    Requires PIXAZO_API_KEY in .env.
    Raises RuntimeError with an actionable message on any failure.
    """
    api_key = os.getenv("PIXAZO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PIXAZO_API_KEY is not set in .env. "
            "Please configure it to enable comic image generation."
        )

    enhanced_prompt = (
        f"Educational comic-book illustration style, colorful educational illustration, "
        f"consistent comic/storybook style, clear characters and environment, "
        f"no copyrighted characters, no unnecessary text inside the image.\n\nDescription: {prompt}"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/json"
    }
    
    data = {
        "prompt": enhanced_prompt
    }

    try:
        response = requests.post(_PIXAZO_URL, headers=headers, json=data)
        
        # Pixazo API might be down or endpoint might have changed if it returns 404
        if response.status_code == 404:
            raise RuntimeError(f"Pixazo API free endpoint unavailable (404 Resource not found).")
            
        response.raise_for_status()
        resp_json = response.json()
        
        # Handle async polling if a request ID is returned instead of direct URL
        if "id" in resp_json and "url" not in resp_json:
            request_id = resp_json["id"]
            status_url = _STATUS_URL_TEMPLATE.format(request_id=request_id)
            
            for _ in range(_MAX_RETRIES):
                time.sleep(_POLL_INTERVAL_SEC)
                status_resp = requests.get(status_url, headers=headers)
                status_resp.raise_for_status()
                status_data = status_resp.json()
                
                if status_data.get("status") == "completed" and "url" in status_data:
                    image_url = status_data["url"]
                    break
                elif status_data.get("status") in ["failed", "error"]:
                    raise RuntimeError(f"Pixazo async generation failed: {status_data}")
            else:
                raise RuntimeError("Pixazo API polling timed out.")
        else:
            image_url = resp_json.get("url") or resp_json.get("image_url")
            if not image_url:
                # If there's an image base64 directly returned
                if "image" in resp_json:
                    import base64
                    return base64.b64decode(resp_json["image"])
                raise RuntimeError(f"Pixazo API did not return an image URL. Response: {resp_json}")
        
        # Download the actual image bytes
        img_resp = requests.get(image_url)
        img_resp.raise_for_status()
        return img_resp.content

    except requests.exceptions.RequestException as e:
        if e.response is not None:
            if e.response.status_code == 401:
                raise RuntimeError("Invalid PIXAZO_API_KEY. Check your API token in .env.")
            if e.response.status_code == 429:
                raise RuntimeError("Pixazo rate limit or quota exceeded (429). Wait a moment and retry.")
        raise RuntimeError(f"Pixazo API error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during image generation: {str(e)[:300]}")
