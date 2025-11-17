from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import pytesseract
from pathlib import Path
import os
import uuid
import time
from werkzeug.utils import secure_filename
import logging
import tempfile

# ---------------------------
# Configure logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')  # <-- ensure it finds index.html

# ---------------------------
# Configuration — Use /tmp for Hugging Face-safe writable dir
# ---------------------------
BASE_DIR = Path(tempfile.gettempdir())
app.config['UPLOAD_FOLDER'] = BASE_DIR / 'uploads'
app.config['PROCESSED_FOLDER'] = BASE_DIR / 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Create directories if not exist
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['PROCESSED_FOLDER'].mkdir(exist_ok=True)

# Tesseract binary path (adjust for your environment)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# ---------------------------
# Allowed file types
# ---------------------------
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------
# Image Processing
# ---------------------------
def deskew_image(image):
    """Detect and correct skew in image."""
    try:
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 10:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle -= 90
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        logger.info(f"Deskewed by {angle:.2f} degrees")
        return rotated
    except Exception as e:
        logger.warning(f"Deskew failed: {e}")
        return image

def enhance_image_quality(image_path):
    """Enhance image before OCR."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Remove blue ink lines
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 40, 40])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        cleaned = cv2.inpaint(img, blue_mask, 3, cv2.INPAINT_TELEA)

        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
        deskewed = deskew_image(denoised)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(deskewed)
        smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        refined = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        final = cv2.bitwise_not(refined)
        return final
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        raise

def recognize_characters(enhanced_image):
    """Perform OCR on enhanced image."""
    configs = [
        "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "--psm 7 --oem 3",
        "--psm 11 --oem 3",
        "--psm 13 --oem 3"
    ]
    best_result, max_confidence = "", 0

    for config in configs:
        try:
            data = pytesseract.image_to_data(enhanced_image, config=config, output_type=pytesseract.Output.DICT)
            characters, total_conf, count = [], 0, 0
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                if text and conf > 20:
                    x, y = data['left'][i], data['top'][i]
                    h = data['height'][i]
                    characters.append((y, x, text, h))
                    total_conf += conf
                    count += 1
            if count == 0:
                continue
            avg_conf = total_conf / count
            characters.sort(key=lambda item: (item[0] // max(item[3], 10), item[1]))
            result = ''.join([ch[2] for ch in characters])
            if avg_conf > max_confidence and len(result) > len(best_result):
                max_confidence, best_result = avg_conf, result
        except Exception as e:
            logger.warning(f"OCR config failed: {e}")
            continue

    try:
        fallback = pytesseract.image_to_string(enhanced_image, config='--psm 6 --oem 3').strip()
        if len(fallback) > len(best_result):
            best_result = fallback
    except Exception as e:
        logger.warning(f"Fallback OCR failed: {e}")
    
    return best_result, max_confidence


# ---------------------------
# File utilities
# ---------------------------
def safe_remove_file(file_path):
    try:
        if file_path.exists():
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Error removing {file_path}: {e}")

def process_single_file(file, file_id):
    """Process and OCR a single image file."""
    upload_path = None
    try:
        original_filename = secure_filename(file.filename)
        file_extension = Path(original_filename).suffix
        upload_path = app.config['UPLOAD_FOLDER'] / f"{file_id}{file_extension}"
        file.save(upload_path)

        enhanced_image = enhance_image_quality(upload_path)
        processed_filename = f"enhanced_{file_id}.png"
        processed_path = app.config['PROCESSED_FOLDER'] / processed_filename
        cv2.imwrite(str(processed_path), enhanced_image)
        recognized_text, confidence = recognize_characters(enhanced_image)
        
        return {
            'success': True,
            'original_filename': original_filename,
            'processed_filename': processed_filename,
            'recognized_text': recognized_text,
            'confidence': confidence,
            'file_id': file_id
        }
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        return {'success': False, 'error': str(e)}
    finally:
        if upload_path and upload_path.exists():
            safe_remove_file(upload_path)


# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        result = process_single_file(file, str(uuid.uuid4()))
        if result['success']:
            return jsonify(result)
        return jsonify({'error': result['error']}), 500
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    """Handle multiple file uploads."""
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'error': 'Empty file list'}), 400

    results = []
    success_count, fail_count = 0, 0

    for file in files:
        if not allowed_file(file.filename):
            results.append({
                'success': False,
                'original_filename': file.filename,
                'error': 'Invalid file type'
            })
            fail_count += 1
            continue

        try:
            file_id = str(uuid.uuid4())
            result = process_single_file(file, file_id)
            if result['success']:
                success_count += 1
            else:
                fail_count += 1
            results.append(result)
        except Exception as e:
            fail_count += 1
            results.append({
                'success': False,
                'original_filename': file.filename,
                'error': str(e)
            })

    return jsonify({
        'success': True,
        'total_files': len(files),
        'successful_processing': success_count,
        'failed_processing': fail_count,
        'results': results
    })


@app.route('/download/<filename>')
def download_file(filename):
    file_path = app.config['PROCESSED_FOLDER'] / secure_filename(filename)
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'upload_folder': str(app.config['UPLOAD_FOLDER']),
        'processed_folder': str(app.config['PROCESSED_FOLDER']),
        'upload_writable': os.access(app.config['UPLOAD_FOLDER'], os.W_OK),
        'processed_writable': os.access(app.config['PROCESSED_FOLDER'], os.W_OK),
        'tesseract_exists': os.path.exists(pytesseract.pytesseract.tesseract_cmd)
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Route not found'}), 404


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == '__main__':
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Processed folder: {app.config['PROCESSED_FOLDER']}")
    logger.info(f"Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
    logger.info("Starting OCR Flask Application...")
    app.run(host='0.0.0.0', port=7860, debug=False)
