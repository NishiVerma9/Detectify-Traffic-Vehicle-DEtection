from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
from detect import run  

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
RESULT_FOLDER = 'results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

           
            result_path = run(file_path)  
            

            return render_template('result.html', result_image=result_path)

if __name__ == '__main__':
    app.run(debug=True)
