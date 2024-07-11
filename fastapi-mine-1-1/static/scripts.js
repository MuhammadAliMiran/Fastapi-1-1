document.addEventListener('DOMContentLoaded', () => {
    const compareFacesForm = document.getElementById('compare-faces-form');
    const openCameraButton1 = document.getElementById('open-camera1');
    const openCameraButton2 = document.getElementById('open-camera2');
    const takePhotoButton1 = document.getElementById('take-photo1');
    const takePhotoButton2 = document.getElementById('take-photo2');
    const camera1 = document.getElementById('camera1');
    const camera2 = document.getElementById('camera2');
    const photoCanvas1 = document.getElementById('photo-canvas1');
    const photoCanvas2 = document.getElementById('photo-canvas2');
    const fileInput1 = document.getElementById('compare-face-file1');
    const fileInput2 = document.getElementById('compare-face-file2');
    const img1 = document.getElementById('img1');
    const img2 = document.getElementById('img2');
    let stream1, stream2;

    async function startCamera(cameraElement, takePhotoButton) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            cameraElement.srcObject = stream;
            cameraElement.style.display = 'block';
            takePhotoButton.style.display = 'block';
            return stream;
        } catch (err) {
            console.error('Error accessing camera: ', err);
        }
    }

    function capturePhoto(cameraElement, canvasElement) {
        const context = canvasElement.getContext('2d');
        canvasElement.width = cameraElement.videoWidth;
        canvasElement.height = cameraElement.videoHeight;
        context.drawImage(cameraElement, 0, 0, canvasElement.width, canvasElement.height);
        return new Promise((resolve) => {
            canvasElement.toBlob((blob) => resolve(blob), 'image/jpeg');
        });
    }

    function stopCamera(stream, cameraElement, takePhotoButton) {
        stream.getTracks().forEach(track => track.stop());
        cameraElement.style.display = 'none';
        takePhotoButton.style.display = 'none';
    }

    openCameraButton1.addEventListener('click', async () => {
        stream1 = await startCamera(camera1, takePhotoButton1);
    });

    openCameraButton2.addEventListener('click', async () => {
        stream2 = await startCamera(camera2, takePhotoButton2);
    });

    takePhotoButton1.addEventListener('click', () => {
        capturePhoto(camera1, photoCanvas1).then((photoBlob1) => {
            const img1 = document.getElementById('img1');
            img1.src = URL.createObjectURL(photoBlob1);
            img1.style.display = 'block';
            // Store the photoBlob1 for form submission
            document.getElementById('compare-face-file1').photoBlob = photoBlob1;
            // Stop the camera after taking the photo
            stopCamera(stream1, camera1, takePhotoButton1);
        });
    });

    takePhotoButton2.addEventListener('click', () => {
        capturePhoto(camera2, photoCanvas2).then((photoBlob2) => {
            const img2 = document.getElementById('img2');
            img2.src = URL.createObjectURL(photoBlob2);
            img2.style.display = 'block';
            // Store the photoBlob2 for form submission
            document.getElementById('compare-face-file2').photoBlob = photoBlob2;
            // Stop the camera after taking the photo
            stopCamera(stream2, camera2, takePhotoButton2);
        });
    });

    fileInput1.addEventListener('change', () => {
        const file = fileInput1.files[0];
        if (file) {
            img1.src = URL.createObjectURL(file);
            img1.style.display = 'block';
        }
    });

    fileInput2.addEventListener('change', () => {
        const file = fileInput2.files[0];
        if (file) {
            img2.src = URL.createObjectURL(file);
            img2.style.display = 'block';
        }
    });

    compareFacesForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData();

        const fileInput1 = document.getElementById('compare-face-file1');
        const fileInput2 = document.getElementById('compare-face-file2');

        if (fileInput1.photoBlob) {
            formData.append('file1', fileInput1.photoBlob, 'photo1.jpg');
        } else {
            formData.append('file1', fileInput1.files[0]);
        }

        if (fileInput2.photoBlob) {
            formData.append('file2', fileInput2.photoBlob, 'photo2.jpg');
        } else {
            formData.append('file2', fileInput2.files[0]);
        }

        try {
            const response = await fetch('/compare_faces/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An error occurred');
            }

            const result = await response.json();
            document.getElementById('result').textContent = `Similarity: ${result.similarity}`;
        } catch (error) {
            alert(`Error: ${error.message}`);
            document.getElementById('result').textContent = '';
        }
    });
});
