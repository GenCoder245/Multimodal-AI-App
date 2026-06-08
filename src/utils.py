import base64
from io import BytesIO
from PIL import Image

def b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)

def pil_image_from_bytes(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b))