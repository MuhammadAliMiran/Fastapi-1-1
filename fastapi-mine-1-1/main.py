from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from pydantic import BaseModel
import numpy as np
import dlib
import cv2
from PIL import Image, UnidentifiedImageError
import io
import logging
from fastapi.staticfiles import StaticFiles
from facenet_pytorch import InceptionResnetV1
import torch
import json
import os

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Initialize FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load dlib's face detector
face_detector = dlib.get_frontal_face_detector()

class FaceMetadata(BaseModel):
    id: str
    name: str

def save_face_image(face_image, filename):
    save_path = os.path.join("static/faces", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    face_pil_image = Image.fromarray(face_image)
    face_pil_image.save(save_path)
    logging.info(f"Saved face image at {save_path}")
    return save_path

def extract_face(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        logging.info(f"Original image mode: {image.mode}")
        image = image.convert("RGB")  # Ensure the image is in RGB format
        logging.info("Image converted to RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail="Error processing image")

    image = np.array(image)
    logging.info(f"Image shape after conversion to array: {image.shape}")
    logging.info(f"Image dtype: {image.dtype}")

    if image.dtype != np.uint8:
        logging.error("Image is not in 8-bit format")
        raise HTTPException(status_code=400, detail="Image must be 8-bit per channel")

    # Detect faces using dlib
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)

    logging.info(f"Number of faces detected: {len(faces)}")
    
    padding_top = 100  # Define the amount of padding you want at the top
    padding_sides = 50  # Define the amount of padding you want on the sides and bottom

    if len(faces) == 1:
        face = faces[0]
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Calculate the coordinates with extra padding at the top and regular padding on other sides
        x_padded = max(x - padding_sides, 0)
        y_padded = max(y - padding_top, 0)
        w_padded = min(x + w + padding_sides, image.shape[1]) - x_padded
        h_padded = min(y + h + padding_sides, image.shape[0]) - y_padded
        
        # Extract the padded face region
        face_image = image[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
        return face_image


    elif len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    else:
        raise HTTPException(status_code=400, detail="Multiple faces detected in the image")

def get_face_embedding(face_image):
    face_image = cv2.resize(face_image, (160, 160))  # Resize image to 160x160 for FaceNet
    face_image = np.transpose(face_image, (2, 0, 1))  # Convert to CHW format
    face_image = torch.tensor(face_image).float().to(device)
    face_image = (face_image - 127.5) / 128.0  # Normalize image
    face_image = face_image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(face_image).cpu().numpy().flatten()
    return embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.get("/")
async def root():
    return {"message": "Face Similarity API"}

@app.post("/compare_faces/")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    contents1 = await file1.read()
    contents2 = await file2.read()

    try:
        face_image1 = extract_face(contents1)
        face_image2 = extract_face(contents2)

        # Save the cropped face images
        save_face_image(face_image1, file1.filename)
        save_face_image(face_image2, file2.filename)

        embedding1 = get_face_embedding(face_image1)
        embedding2 = get_face_embedding(face_image2)
    except HTTPException as e:
        logging.error(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    similarity = cosine_similarity(embedding1, embedding2)

    # Round similarity to 2 decimal places and ensure it's a Python float
    similarity = float(round(similarity, 2))

    # Prepare response as a dictionary
    response = {
        "similarity": similarity,
        "image1": file1.filename,
        "image2": file2.filename
    }

    # Serialize the response dictionary to JSON
    json_response = json.dumps(response)

    # Return the JSON response with proper content type
    return Response(content=json_response, media_type="application/json")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
