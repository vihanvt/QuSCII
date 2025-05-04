import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import BackgroundTasks 
import os
from PIL import Image, ImageDraw, ImageFont
from math import pi
from pathlib import Path
import shutil
import uuid
from functools import lru_cache

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
#using jinja to include html with flask
templates = Jinja2Templates(directory="templates")
Path("uploads").mkdir(exist_ok=True)
Path("static/results").mkdir(exist_ok=True)

def quantum(inp_path, out_path, block_size, font_size, magnitude, ascii_set, use_color=True):
    brightness_cache = {}
    
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
        simulator = Aer.get_backend('qasm_simulator')
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

    for x in range(rows):
        for y in range(cols):
            r, g, b = resized.convert("RGB").getpixel((y, x))
            r_new = get_brightness(r)
            g_new = get_brightness(g)
            b_new = get_brightness(b)
            fill = (r_new, g_new, b_new)
            bright = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
            ascii_index = min(int((bright / 255) * (len(ascii_set) - 1)), len(ascii_set) - 1)
            char = ascii_set[ascii_index]
            draw.text(
                (y * char_width, x * char_height),char,font=font,fill=fill)
    
    output = output.convert('RGB')
    output.save(out_path)
    return output

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

job_status={}

@app.post("/upload")
async def upload_file(request: Request,background_tasks:BackgroundTasks, file: UploadFile = File(...), magnitude: float = Form(...)):
    process_id = str(uuid.uuid4())
    file_path = f"uploads/{process_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_status[process_id]="queued"
    background_tasks.add_task(
        process_image,
        process_id=process_id,
        file_path=file_path,
        magnitude=magnitude,
    )

    return templates.TemplateResponse(
        "loading.html", 
        {
            "request": request,
            "processing_id": process_id,
            "magnitude": magnitude,
            "filename": file.filename
        }
    )

async def process_image(process_id: str, file_path: str, magnitude: float):
    out_path = f"static/results/{process_id}_result.jpg"
    ascii_set = ".:-=+*#%@"
    try:
        job_status[process_id] = "processing"
        quantum(
            inp_path=file_path,
            out_path=out_path,
            block_size=6,
            font_size=10,
            magnitude=magnitude,
            ascii_set=ascii_set,
            use_color=True
        )
        job_status[process_id] = "completed"
    except Exception as e:
        job_status[process_id] = f"error: {str(e)}"

@app.get("/process/{processing_id}")
async def check_process_status(processing_id: str):
    result_path = f"static/results/{processing_id}_result.jpg"

    if os.path.exists(result_path):
        job_status[processing_id] = "completed"
        return {"status": "completed"}

    return {"status": job_status.get(processing_id, "not_found")}

@app.get("/result", response_class=HTMLResponse)
async def result_page(request: Request, process_id: str):
    result_path = f"static/results/{process_id}_result.jpg"

    if not os.path.exists(result_path):
        return RedirectResponse(url="/", status_code=303)
    
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "process_id": process_id
        }
    )

@app.get("/image/{process_id}")
async def get_image(process_id: str):
    image_path = f"static/results/{process_id}_result.jpg"
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    return FileResponse(image_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
