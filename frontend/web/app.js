class FaceSwapApp {
    constructor() {
        this.video = document.getElementById('videoElement');
        this.canvas = document.getElementById('outputCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        this.isProcessing = false;
        this.selectedFile = null;
        this.processingInterval = null;

        // Face tracking and stability
        this.frameBuffer = [];
        this.bufferSize = 3;
        this.skipFrames = 0;
        this.processEveryNFrames = 2;
        this.lastSuccessfulFrame = null;

        // API endpoints
        this.AI_SERVICE_URL = 'http://localhost:8000';
        this.API_SERVER_URL = 'http://localhost:3001';

        this.initializeEventListeners();
        this.checkAIServiceStatus();
    }

    initializeEventListeners() {
        // Button listeners
        document.getElementById('startBtn').addEventListener('click', () => this.startCamera());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopCamera());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearReferenceFace());
        document.getElementById('uploadBtn').addEventListener('click', () => this.uploadReferenceFace());

        // File upload listeners
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleFileDrop(e));

        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
    }

    async checkAIServiceStatus() {
        try {
            const response = await fetch(`${this.AI_SERVICE_URL}/health`);
            const data = await response.json();

            if (data.status === 'OK') {
                this.updateStatus('aiStatus', 'aiStatusText', true, 'Connected');
                this.checkReferenceStatus();
            } else {
                throw new Error('Service unavailable');
            }
        } catch (error) {
            console.error('AI Service connection failed:', error);
            this.updateStatus('aiStatus', 'aiStatusText', false, 'Disconnected');
        }
    }

    async checkReferenceStatus() {
        try {
            const response = await fetch(`${this.AI_SERVICE_URL}/reference-status`);
            const data = await response.json();

            if (data.is_loaded) {
                this.updateStatus('faceStatus', 'faceStatusText', true, 'Loaded');
                document.getElementById('clearBtn').disabled = false;
            } else {
                this.updateStatus('faceStatus', 'faceStatusText', false, 'Not Loaded');
            }
        } catch (error) {
            console.error('Failed to check reference status:', error);
        }
    }

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: 480,
                    height: 360,
                    facingMode: 'user'
                }
            });

            this.video.srcObject = this.stream;

            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.canvas.style.display = 'block';

                this.updateStatus('cameraStatus', 'cameraStatusText', true, 'Active');
                document.getElementById('videoStatus').textContent = 'Camera Active';
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;

                // Start processing frames
                this.startFrameProcessing();
            };

        } catch (error) {
            console.error('Error starting camera:', error);
            alert('Failed to start camera. Please check permissions.');
            this.updateStatus('cameraStatus', 'cameraStatusText', false, 'Error');
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        this.video.srcObject = null;
        this.canvas.style.display = 'none';

        this.stopFrameProcessing();

        this.updateStatus('cameraStatus', 'cameraStatusText', false, 'Stopped');
        this.updateStatus('swapStatus', 'swapStatusText', false, 'Inactive');

        document.getElementById('videoStatus').textContent = 'Camera Stopped';
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    }

    startFrameProcessing() {
        this.processingInterval = setInterval(() => {
            this.processFrame();
        }, 33); // Process at ~30 FPS but skip frames for performance
    }

    stopFrameProcessing() {
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
    }

    async processFrame() {
        if (this.isProcessing || !this.video.videoWidth) {
            return;
        }

        // Skip frames for better performance and stability
        this.skipFrames++;
        if (this.skipFrames < this.processEveryNFrames) {
            if (this.lastSuccessfulFrame) {
                // Display last successful frame while skipping
                const img = new Image();
                img.onload = () => {
                    this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
                };
                img.src = this.lastSuccessfulFrame;
            } else {
                // Show original video if no successful frame yet
                this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            }
            return;
        }
        this.skipFrames = 0;

        this.isProcessing = true;

        try {
            // Draw video frame to canvas with dynamic sizing
            this.canvas.width = this.video.videoWidth || 640;
            this.canvas.height = this.video.videoHeight || 480;
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

            // Get frame data with higher quality for better face detection
            const frameData = this.canvas.toDataURL('image/jpeg', 0.85);

            // Send to AI service with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000);

            const response = await fetch(`${this.AI_SERVICE_URL}/process-frame`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    frame_data: frameData
                }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);
            const result = await response.json();

            if (result.success && result.frame) {
                // Store successful frame for frame skipping
                this.lastSuccessfulFrame = result.frame;

                // Display processed frame
                const img = new Image();
                img.onload = () => {
                    this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
                };
                img.src = result.frame;

                // Update swap status with detailed info
                if (result.face_swapped) {
                    this.updateStatus('swapStatus', 'swapStatusText', true, 'Active - Tracking');
                    document.getElementById('videoStatus').textContent = result.message || 'Face Swap Active';
                } else {
                    this.updateStatus('swapStatus', 'swapStatusText', false, 'Searching...');
                    document.getElementById('videoStatus').textContent = result.message || 'Looking for face...';
                }
            } else {
                // Fallback to original video
                this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            }

        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Frame processing timeout - using fallback');
            } else {
                console.error('Frame processing error:', error);
            }

            // Use last successful frame or original video on error
            if (this.lastSuccessfulFrame) {
                const img = new Image();
                img.onload = () => {
                    this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
                };
                img.src = this.lastSuccessfulFrame;
            } else {
                this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            }

            this.updateStatus('swapStatus', 'swapStatusText', false, 'Processing Error');
        }

        this.isProcessing = false;
    }

    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
    }

    handleFileDrop(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        // Validate file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            alert('File size must be less than 5MB.');
            return;
        }

        this.selectedFile = file;
        this.displayPreview(file);
        document.getElementById('uploadBtn').disabled = false;
    }

    displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewContainer = document.getElementById('previewContainer');
            previewContainer.innerHTML = `
                <img src="${e.target.result}" alt="Preview" class="preview-image">
                <p style="margin-top: 10px; font-size: 0.9rem; color: #666;">
                    ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)
                </p>
            `;
        };
        reader.readAsDataURL(file);
    }

    async uploadReferenceFace() {
        if (!this.selectedFile) {
            alert('Please select an image first.');
            return;
        }

        const uploadBtn = document.getElementById('uploadBtn');
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Uploading...';

        try {
            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    const response = await fetch(`${this.AI_SERVICE_URL}/upload-reference-face`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            image_data: e.target.result
                        })
                    });

                    const result = await response.json();

                    if (result.success) {
                        this.updateStatus('faceStatus', 'faceStatusText', true, 'Loaded');
                        document.getElementById('clearBtn').disabled = false;
                        alert('Reference face uploaded successfully!');
                    } else {
                        throw new Error(result.detail || 'Upload failed');
                    }

                } catch (error) {
                    console.error('Upload error:', error);
                    alert(`Upload failed: ${error.message}`);
                }

                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Upload Face';
            };

            reader.readAsDataURL(this.selectedFile);

        } catch (error) {
            console.error('File reading error:', error);
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload Face';
        }
    }

    async clearReferenceFace() {
        try {
            const response = await fetch(`${this.AI_SERVICE_URL}/clear-reference`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (result.success) {
                this.updateStatus('faceStatus', 'faceStatusText', false, 'Cleared');
                this.updateStatus('swapStatus', 'swapStatusText', false, 'Inactive');
                document.getElementById('clearBtn').disabled = true;
                document.getElementById('previewContainer').innerHTML = '';
                document.getElementById('fileInput').value = '';
                this.selectedFile = null;
                alert('Reference face cleared successfully!');
            }

        } catch (error) {
            console.error('Clear reference error:', error);
            alert('Failed to clear reference face.');
        }
    }

    updateStatus(indicatorId, textId, isActive, text) {
        const indicator = document.getElementById(indicatorId);
        const statusText = document.getElementById(textId);

        indicator.className = `status-indicator ${isActive ? 'status-active' : 'status-inactive'}`;
        statusText.textContent = text;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FaceSwapApp();
});