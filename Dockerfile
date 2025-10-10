FROM llama_cpp_cuda:latest

WORKDIR /app

# Install dependencies for the application
RUN apt-get install -y poppler-utils tesseract-ocr libtesseract-dev ffmpeg libsm6 libxext6

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set entrypoint
ENTRYPOINT ["python", "run.py"]
