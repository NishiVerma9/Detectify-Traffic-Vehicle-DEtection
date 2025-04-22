from flask import Flask, render_template, request, jsonify, url_for, send_file
from flask_cors import CORS
import os
import shutil
from werkzeug.utils import secure_filename
import torch
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import cv2
import tempfile
from pathlib import Path
import numpy as np

# Initialize YOLO model
try:
    device = select_device('')  # Use GPU if available, else CPU
    model = DetectMultiBackend('yolov5s.pt', device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)
    model.warmup(imgsz=(1, 3, *imgsz))  
except Exception as e:
    print(f"Error loading model: {e}")
    raise

app = Flask(__name__, 
    static_url_path='',
    static_folder='static',
    template_folder='templates')
CORS(app)

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
TEMP_FOLDER = os.path.join(BASE_DIR, 'runs', 'detect')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  


for folder in [UPLOAD_FOLDER, RESULT_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error deleting {file_path}: {e}')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename))

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Check if file is video
            is_video = filename.lower().endswith(('.mp4', '.avi', '.mov'))
            
            if is_video:
                try:
                    cap = cv2.VideoCapture(file_path)
                    if not cap.isOpened():
                        raise Exception("Error opening video file")
                        
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Always save as MP4 for better browser compatibility
                    result_filename = f'result_{os.path.splitext(filename)[0]}.mp4'
                    result_path = os.path.join(RESULT_FOLDER, result_filename)
                    
                   
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*'H264')
                    except:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                        except:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    
                    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
                    
                    detections = []
                    processed_frames = 0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        processed_frames += 1
                        if processed_frames % 2 != 0:  
                            continue
                        
                        # Process frame with YOLO
                        img = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
                        img = img.transpose((2, 0, 1))[::-1]
                        img = np.ascontiguousarray(img)
                        img = torch.from_numpy(img).to(device)
                        img = img.float()
                        img /= 255
                        if len(img.shape) == 3:
                            img = img[None]

                        pred = model(img, augment=False, visualize=False)
                        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
                        
                        for i, det in enumerate(pred):
                            if len(det):
                                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                                for *xyxy, conf, cls in reversed(det):
                                    c = int(cls)
                                    label = f'{names[c]} {conf:.2f}'
                                    detections.append({
                                        'class': names[c],
                                        'confidence': float(conf),
                                        'frame': processed_frames
                                    })
                                    
                                    # Draw detection box
                                    cv2.rectangle(frame, 
                                                (int(xyxy[0]), int(xyxy[1])), 
                                                (int(xyxy[2]), int(xyxy[3])), 
                                                (0, 255, 0), 2)
                                    cv2.putText(frame, label, 
                                              (int(xyxy[0]), int(xyxy[1])-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        out.write(frame)
                        
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    
                    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                        return jsonify({
                            'status': 'success',
                            'message': f'Video processed successfully! ({processed_frames} frames)',
                            'result_path': f'/results/{result_filename}',
                            'original_path': f'/uploads/{filename}',
                            'detections': list({(d['class'], d['confidence']): d for d in detections}.values()),
                            'total_frames': total_frames,
                            'processed_frames': processed_frames
                        })
                    else:
                        raise Exception("Failed to create output video")
                        
                except Exception as e:
                    print(f"Error processing video: {str(e)}")
                    raise
            
            else:
                # Image processing code (unchanged)
                img0 = cv2.imread(file_path)
                img = letterbox(img0, imgsz, stride=stride, auto=pt)[0]
                img = img.transpose((2, 0, 1))[::-1]
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.float()
                img /= 255
                if len(img.shape) == 3:
                    img = img[None]

                pred = model(img, augment=False, visualize=False)
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

                detections = []
                for i, det in enumerate(pred):
                    if len(det):
                        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            label = f'{names[c]} {conf:.2f}'
                            detections.append({
                                'class': names[c],
                                'confidence': float(conf),
                                'bbox': [float(x) for x in xyxy]
                            })
                            cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                            cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                result_filename = f'result_{filename}'
                result_path = os.path.join(RESULT_FOLDER, result_filename)
                cv2.imwrite(result_path, img0)

                return jsonify({
                    'status': 'success',
                    'message': 'Detection completed successfully!',
                    'result_path': f'/results/{result_filename}',
                    'original_path': f'/uploads/{filename}',
                    'detections': detections
                })

        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Detection failed: {str(e)}'
            }), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    port = 5000
    print(f"Starting server on http://localhost:{port}")
    app.run(debug=True, host='127.0.0.1', port=port)