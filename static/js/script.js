document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file');
    const loadingMessage = document.getElementById('loading-message');
    const filePreview = document.getElementById('file-preview');
    const videoPreview = document.getElementById('video-preview');
    const result = document.getElementById('result');

    function updateLoadingMessage(message) {
        loadingMessage.innerHTML = `
            <i class="fas fa-spinner fa-spin"></i>
            ${message}
        `;
        loadingMessage.style.display = 'block';
    }

    // Preview file when selected
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            if (file.type.startsWith('video/')) {
                filePreview.style.display = 'none';
                videoPreview.src = URL.createObjectURL(file);
                videoPreview.style.display = 'block';
            } else if (file.type.startsWith('image/')) {
                videoPreview.style.display = 'none';
                const reader = new FileReader();
                reader.onload = function(e) {
                    filePreview.src = e.target.result;
                    filePreview.style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        }
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) return;

        updateLoadingMessage('Uploading file...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.status === 'error') {
                throw new Error(data.message);
            }

            updateLoadingMessage(data.message || 'Processing...');

            const resultImage = document.getElementById('result-image');
            const resultVideo = document.getElementById('result-video');
            const downloadLink = document.getElementById('avi-download-link');

            resultImage.style.display = 'none';
            resultVideo.style.display = 'none';
            downloadLink.style.display = 'none';

            if (file.type.startsWith('video/')) {
                resultVideo.src = data.result_path;
                resultVideo.style.display = 'block';
                downloadLink.href = data.result_path;
                downloadLink.style.display = 'block';
            } else {
                resultImage.src = data.result_path;
                resultImage.style.display = 'block';
            }

            // Show detection list
            const detectionsList = document.getElementById('detections-list');
            if (data.detections && data.detections.length > 0) {
                detectionsList.innerHTML = '<h3>Detected Objects:</h3>' + data.detections.map(
                    det => `<p>${det.class} - Confidence: ${(det.confidence * 100).toFixed(2)}%</p>`
                ).join('');
                detectionsList.style.display = 'block';
            } else {
                detectionsList.style.display = 'none';
            }

        } catch (error) {
            console.error('Error:', error);
            result.innerHTML += `
                <div class="error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    ${error.message}
                </div>
            `;
        } finally {
            loadingMessage.style.display = 'none';
        }
    });

    // Modal handling
    const loginBtn = document.getElementById('loginBtn');
    const signupBtn = document.getElementById('signupBtn');
    const loginModal = document.getElementById('loginModal');
    const signupModal = document.getElementById('signupModal');
    const closeBtns = document.querySelectorAll('.close, .close-signup');

    loginBtn.onclick = () => loginModal.classList.add('active');
    signupBtn.onclick = () => signupModal.classList.add('active');

    closeBtns.forEach(btn => {
        btn.onclick = function() {
            loginModal.classList.remove('active');
            signupModal.classList.remove('active');
        }
    });
});