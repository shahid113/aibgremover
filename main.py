from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from rembg import remove
import io
import logging
import traceback
import sys
import time
from typing import Optional, Tuple
import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
import hashlib
from datetime import datetime
import uvicorn
import onnxruntime
import multipart

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d'
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FORMATS = {"image/jpeg", "image/png"}
CHUNK_SIZE = 8192  # 8KB chunks for streaming
MAX_CONCURRENT_REQUESTS = 5

app = FastAPI(
    title="Background Removal API",
    description="API for removing image backgrounds with optimization and safety features",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Process pool for CPU-intensive tasks
process_pool = ProcessPoolExecutor(max_workers=1)  # Limit to 1 worker for single vCPU

# Semaphore for limiting concurrent requests
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def remove_background_from_image(image_bytes: bytes, hd: bool) -> bytes:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        output = remove(image, alpha_matting=hd)
        
        output_bytes = io.BytesIO()
        output.save(output_bytes, format="PNG", optimize=True)
        return output_bytes.getvalue()
    except Exception as e:
        logger.error(f"Background removal error: {str(e)}\n{traceback.format_exc()}")
        raise

class ImageProcessor:
    @staticmethod
    def validate_image(image_data: bytes) -> bool:
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                img.verify()
            return True
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            return False

async def process_image(image_data: bytes, hd: bool) -> bytes:
    try:
        loop = asyncio.get_event_loop()
        
        logger.debug(f"Processing image of size: {len(image_data)} bytes, HD mode: {hd}")
        
        if not ImageProcessor.validate_image(image_data):
            raise ValueError("Invalid image data")

        with Image.open(io.BytesIO(image_data)) as image:
            logger.debug(f"Image mode: {image.mode}, size: {image.size}")

        return await loop.run_in_executor(process_pool, remove_background_from_image, image_data, hd)
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}\n{traceback.format_exc()}")
        raise

@app.post("/remove-background/")
async def remove_background(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    hd: bool = False
):
    start_time = time.time()
    request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    logger.info(f"Request {request_id} started - File: {file.filename}, Size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    try:
        async with request_semaphore:
            logger.debug(f"Request {request_id} - Content type: {file.content_type}")
            if file.content_type not in SUPPORTED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.content_type}. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
                )

            image_data = bytearray()
            total_bytes = 0
            
            try:
                while chunk := await file.read(CHUNK_SIZE):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large: {total_bytes/1024/1024:.2f}MB. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
                        )
                    image_data.extend(chunk)
            except Exception as e:
                logger.error(f"Request {request_id} - File reading error: {str(e)}\n{traceback.format_exc()}")
                raise HTTPException(status_code=400, detail=f"File reading error: {str(e)}")

            if not image_data:
                raise HTTPException(status_code=400, detail="Empty file")

            logger.debug(f"Request {request_id} - File size: {total_bytes/1024:.2f}KB")

            try:
                processed_image = await process_image(bytes(image_data), hd)
                output_bytes = io.BytesIO(processed_image)
            except Exception as e:
                logger.error(f"Request {request_id} - Processing error: {str(e)}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

            processing_time = time.time() - start_time
            logger.info(f"Request {request_id} - Processing completed in {processing_time:.2f} seconds")

            headers = {
                "Content-Disposition": "inline; filename=output.png",
                "X-Processing-Time": str(processing_time),
                "X-Request-ID": request_id,
                "Cache-Control": "public, max-age=86400"
            }
            
            return StreamingResponse(
                io.BytesIO(output_bytes.getvalue()),
                media_type="image/png",
                headers=headers
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request {request_id} - Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    