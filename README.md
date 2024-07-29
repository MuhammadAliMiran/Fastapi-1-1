# Face Similarity API

This project provides a Face Similarity API built using FastAPI, dlib, OpenCV, PIL, and FaceNet. It allows users to compare the similarity between two face images and returns a similarity score.

## Features

- **Face Detection**: Detects faces in uploaded images using dlib's face detector.
- **Face Cropping and Padding**: Crops faces from the images with padding to ensure the faces are properly centered.
- **Face Embedding Extraction**: Uses FaceNet to extract face embeddings for similarity comparison.
- **Cosine Similarity Calculation**: Calculates the cosine similarity between the face embeddings of two images.
- **Web Interface**: Provides a simple web interface to upload images or take photos using the camera for comparison.

# Usage

## Web Interface

- **Compare Faces**: Upload two images or take photos using the camera, then click "Compare Faces" to get the similarity score between the two faces.

## API Endpoints

- **Root Endpoint**: GET /
  Returns a welcome message.

## Compare Faces Endpoint: POST /compare_faces/

- **Parameters**:
  - **file1**: The first image file.
  - **file2**: The second image file.

## Returns:

- **similarity**: The similarity score between the two faces.
- **image1**: The filename of the first image.
- **image2**: The filename of the second image.

# High-level Code Overview

## main.py

- **Imports and Initial Setup**: Imports necessary libraries and initializes the FastAPI app, logging, and machine learning model (FaceNet).
- **Face Detection**: Uses dlib to detect faces in images.
- **Face Cropping and Padding**: Crops the detected face with padding to ensure proper centering.
- **Face Embedding Extraction**: Converts the face image to a format suitable for FaceNet and extracts embeddings.
- **Cosine Similarity Calculation**: Computes the cosine similarity between two face embeddings.

## index.html

- Provides a user interface to upload images or take photos for face comparison.

## scripts.js

- Handles the frontend logic for capturing images from the camera and submitting the form to the API.

## styles.css

- Contains the styling for the web interface.
