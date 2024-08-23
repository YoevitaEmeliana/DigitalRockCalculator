from flask import Flask, render_template, request
import zipfile
import os
import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi
import pandas as pd
import re
import shutil
from skimage import morphology
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt


app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Use a more secure key for production
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'zip'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def calculate_specific_area(file_path):
    try:
        data = sio.loadmat(file_path)
        var_names = [name for name in data if not name.startswith('__')]
        
        if not var_names:
            return None
        
        M = data[var_names[0]]
        
        if M.ndim == 3 and M.shape == (128, 128, 128):
            A = M[:, :, 127]  # Use 127 for a valid index
            A = (A > 0).astype(np.float64)
            
            resolution = 2.25  # µm/pixel
            A = morphology.binary_opening(A, morphology.disk(1))  # Disk of radius 1
            
            # Calculate perimeter using dilation and XOR
            A_dilated = ndi.binary_dilation(A)
            P = np.sum(A_dilated ^ A)  # Perimeter calculation
            
            At = A.size
            Ag = np.sum(A)
            Ap = np.sum(~A)
            
            # Specific surface area calculations
            Specific_surface_of_pores_2D = P / (resolution * Ap) * 1000  # mm^-1
            Specific_surface_of_pores_3D = P / (resolution * Ap) * 1.35 * 1000  # mm^-1
            Specific_surface_of_grains_2D = P / (resolution * Ag) * 1000  # mm^-1
            Specific_surface_of_grains_3D = P / (resolution * Ag) * 1.35 * 1000  # mm^-1
            Porosity = Ap / At
            
            return (Specific_surface_of_pores_2D, Specific_surface_of_pores_3D,
                    Specific_surface_of_grains_2D, Specific_surface_of_grains_3D,
                    Porosity)
        else:
            print(f"Data dalam {file_path} tidak memiliki dimensi yang diharapkan.")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def calculate_porosity(file_path):
    try:
        data = sio.loadmat(file_path)
        var_names = [name for name in data if not name.startswith('__')]
        
        if not var_names:
            return None
        
        M = data[var_names[0]]
        
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

def calculate_pore_size(file_path):
    try:
        data = sio.loadmat(file_path)
        var_names = [name for name in data if not name.startswith('__')]
        
        if not var_names:
            return None
        
        M = data[var_names[0]]
        
        if M.ndim == 3 and M.shape == (128, 128, 128):
            A = M[:, :, 127]  # Use 127 for a valid index
            A = (A > 0).astype(np.float64)
            resolution = 2.25  # µm/pixel
            D = -distance_transform_edt(A)
            B = ndi.median_filter(D, size=3)
            B = watershed(B, markers=ndi.label(A)[0], connectivity=2)
            Pr = np.zeros_like(A)
            Pr[A == 0] = 1
            Pr = morphology.binary_opening(Pr, morphology.disk(3))
            labeled, num_features = ndi.label(Pr)
            V = np.bincount(labeled.ravel())[1:]  # Exclude the background
            R = resolution * np.sqrt(V / np.pi)
            R = R[R > 0]
            if len(R) == 0:
                return (0, 0)  # Return zeros if no pores detected
            return (np.mean(R), np.std(R))
        else:
            print(f"Data dalam {file_path} tidak memiliki dimensi yang diharapkan.")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def extract_number_from_filename(filename):
    match = re.search(r'_(\d+)\.', filename)
    return int(match.group(1)) if match else float('inf')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        calculation_type = request.form.get('calculation_type')
        
        if file and allowed_file(file.filename):
            # Clear the upload folder before saving new files
            if os.path.exists(app.config['UPLOAD_FOLDER']):
                shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.makedirs(app.config['UPLOAD_FOLDER'])

            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(zip_path)
            extract_zip(zip_path, app.config['UPLOAD_FOLDER'])
            
            results_porosity = []
            results_poresize = []
            results_specific_area = []
            mat_files = []
            
            for root, _, files in os.walk(app.config['UPLOAD_FOLDER']):
                for file_name in files:
                    if file_name.endswith('.mat'):
                        mat_files.append(os.path.join(root, file_name))
            
            mat_files.sort(key=lambda f: extract_number_from_filename(os.path.basename(f)))
            
            for file_path in mat_files:
                if calculation_type == 'porosity':
                    porosity = calculate_porosity(file_path)
                    if porosity is not None:
                        results_porosity.append((os.path.basename(file_path), porosity))
                
                elif calculation_type == 'pore_size':
                    pore_data = calculate_pore_size(file_path)
                    if pore_data is not None:
                        avg_radius, std_dev = pore_data
                        results_poresize.append((os.path.basename(file_path), avg_radius, std_dev))
                
                elif calculation_type == 'specific_area':
                    specific_area_data = calculate_specific_area(file_path)
                    if specific_area_data is not None:
                        Specific_surface_of_pores_2D, Specific_surface_of_pores_3D, \
                        Specific_surface_of_grains_2D, Specific_surface_of_grains_3D, \
                        Porosity = specific_area_data
                        results_specific_area.append({
                            'File Name': os.path.basename(file_path),
                            'Specific surface of pores 2D (mm^-1)': Specific_surface_of_pores_2D,
                            'Specific surface of pores 3D (mm^-1)': Specific_surface_of_pores_3D,
                            'Specific surface of grains 2D (mm^-1)': Specific_surface_of_grains_2D,
                            'Specific surface of grains 3D (mm^-1)': Specific_surface_of_grains_3D,
                            'Porosity (ratio)': Porosity
                        })


            # Convert results to DataFrame
            df_porosity = pd.DataFrame(results_porosity, columns=['File Name', 'Porosity'])
            df_poresize = pd.DataFrame(results_poresize, columns=['File Name', 'Average Pore Radius (micron)', 'Standard Deviation (micron)'])
            df_specific_area = pd.DataFrame(results_specific_area)
            
            # Save results as CSV files
            csv_path_porosity = os.path.join(app.config['UPLOAD_FOLDER'], 'results_porosity.csv')
            csv_path_poresize = os.path.join(app.config['UPLOAD_FOLDER'], 'results_poresize.csv')
            csv_path_specific_area = os.path.join(app.config['UPLOAD_FOLDER'], 'results_specific_area.csv')
            df_porosity.to_csv(csv_path_porosity, index=False)
            df_poresize.to_csv(csv_path_poresize, index=False)
            df_specific_area.to_csv(csv_path_specific_area, index=False)
            
            tables = {}
            if calculation_type == 'porosity':
                tables["Hasil Perhitungan Porositas"] = df_porosity.to_html(classes='data', index=False)
            elif calculation_type == 'pore_size':
                tables["Hasil Perhitungan Ukuran Pori"] = df_poresize.to_html(classes='data', index=False)
            elif calculation_type == 'specific_area':
                tables["Hasil Perhitungan Specific Area"] = df_specific_area.to_html(classes='data', index=False)
            
            return render_template('index.html', tables=tables)
    
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
