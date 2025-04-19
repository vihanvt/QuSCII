import numpy as np
import time
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
from PIL import Image, ImageDraw, ImageFont
import math
from math import pi

app = Flask(__name__)

brightness_cache = {}
def quantum(inp_path, out_path, block_size, font_size, magnitude, ascii_set, use_color=True):
    global brightness_cache
    brightness_cache = {}  

    try:
        input_image = Image.open(inp_path)
    except FileNotFoundError:
        print("Image not found.")
        return

    input_width, input_height = input_image.size
    rows = input_height // block_size
    cols = input_width // block_size

    resized = input_image.resize((cols, rows), Image.Resampling.LANCZOS)
    font = ImageFont.load_default()

    bbox = font.getbbox("A")
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]

    output_width = cols * char_width
    output_height = rows * char_height
    output = Image.new("RGB", (output_width, output_height), "black")
    draw = ImageDraw.Draw(output)
    simulator = Aer.get_backend('qasm_simulator')

    def get_brightness(val):
        if val in brightness_cache:
            return brightness_cache[val]

        num_qubits = 2
        qc = QuantumCircuit(num_qubits, num_qubits)
        theta = (val / 255.0) * (pi / 2)
        qc.ry(theta, 0)
        qc.ry(theta, 1)

        if magnitude > 0:
            qc.h(0)
            qc.cry((val / 255.0) * (pi / 4), 0, 1)

        qc.measure(range(num_qubits), range(num_qubits))
        final = transpile(qc, simulator)
        result = simulator.run(final, shots=1).result()
        counts = result.get_counts(qc)

        if counts:
            key = list(counts.keys())[0]
            qval = int(key, 2) / (2**num_qubits - 1)
            qbright = qval * 255
            blended = int(val * (1 - magnitude) + qbright * magnitude)
            brightness_cache[val] = blended
            return blended
        return val

    for x in range(rows):
        for y in range(cols):
            r, g, b = resized.convert("RGB").getpixel((y,x))
            if use_color:
                r_new = get_brightness(r)
                g_new = get_brightness(g)
                b_new = get_brightness(b)
                fill = (r_new, g_new, b_new)
                bright = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
            else:
                gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
                bright = get_brightness(gray)
                fill = (bright, bright, bright)
            ascii_index = min(int((bright / 255) * (len(ascii_set) - 1)), len(ascii_set) - 1)
            char = ascii_set[ascii_index]
            draw.text((y * char_width, x * char_height), char, font=font, fill=fill)
            
    output = output.convert('RGB')
    output.save(out_path)
    print("Saved your desired image at:", out_path)
    return output

@app.route('/')
def home():
    return render_template('index.html')  # Renders the home page (index.html)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    inp_path = 'uploads/' + file.filename
    out_path = 'static/result.jpg'
    file.save(inp_path)
    
    magnitude = float(request.form['magnitude'])  
    print(f"Received magnitude value: {magnitude}")  
    
    ascii_set = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    quantum(
        inp_path=inp_path,
        out_path=out_path,
        block_size=6,          
        font_size=10,          
        magnitude=magnitude,  
        ascii_set=ascii_set,
        use_color=True         
    )

    return redirect(url_for('result_page'))

@app.route('/result')
def result_page():
    image_url = url_for('static', filename='result.jpg')  
    return render_template('result.html', image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
