from flask import Flask, render_template, request
import zipfile
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'zip'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def calculate_porosity(file_path):
    try:
        # Muat data dari file .mat
        data = sio.loadmat(file_path)
        
        # Ambil nama variabel dari file .mat
        var_names = [name for name in data if not name.startswith('__')]
        
        if not var_names:
            return None
        
        # Ambil data matriks dari variabel pertama
        M = data[var_names[0]]
        
        # Pastikan data adalah array 3D dengan dimensi yang benar
        if M.ndim == 3 and M.shape == (128, 128, 128):
            matrix_sum = np.sum(M)
            porosity = 1 - (matrix_sum / (128 * 128 * 128))
            return porosity
        else:
            print(f"Data dalam {file_path} tidak memiliki dimensi yang diharapkan.")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def extract_number_from_filename(filename):
    # Menggunakan regex untuk mengekstrak angka dari nama file
    match = re.search(r'_(\d+)\.', filename)
    return int(match.group(1)) if match else float('inf')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(zip_path)
            extract_zip(zip_path, app.config['UPLOAD_FOLDER'])
            
            results = []
            mat_files = []
            
            for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
                for file_name in files:
                    if file_name.endswith('.mat'):
                        mat_files.append(file_name)
            
            # Urutkan file berdasarkan angka di dalam nama file
            mat_files.sort(key=extract_number_from_filename)
            
            for file_name in mat_files:
                file_path = os.path.join(root, file_name)
                porosity = calculate_porosity(file_path)
                if porosity is not None:
                    results.append((file_name, porosity))
            
            # Convert results to a DataFrame
            df = pd.DataFrame(results, columns=['File Name', 'Porosity'])
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
            df.to_csv(csv_path, index=False)
            
            return render_template('index.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
    
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
