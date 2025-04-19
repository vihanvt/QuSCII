import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import os
from PIL import Image, ImageDraw, ImageFont
import math
from math import pi
from pathlib import Path
import shutil
from fastapi import Request

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Ensure directories exist
Path("uploads").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), magnitude: float = Form(...)):
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process image
    out_path = "static/result.jpg"
    ascii_set = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    
    # Return processing page response
    return templates.TemplateResponse(
        "loading.html", 
        {
            "request": request,
            "processing_id": file.filename  # Can be used to track processing status
        }
    )

@app.get("/process/{processing_id}")
async def process_image(request: Request, processing_id: str):
    # Process the image here
    file_path = f"uploads/{processing_id}"
    out_path = "static/result.jpg"
    ascii_set = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    
    # Get magnitude from stored value (you might want to implement a proper state management)
    magnitude = 0.5  # Default value, implement proper state management for actual value
    
    quantum(
        inp_path=file_path,
        out_path=out_path,
        block_size=6,
        font_size=10,
        magnitude=magnitude,
        ascii_set=ascii_set,
        use_color=True
    )
    
    return {"status": "completed"}

@app.get("/result", response_class=HTMLResponse)
async def result_page(request: Request):
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "image_url": "/static/result.jpg"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
