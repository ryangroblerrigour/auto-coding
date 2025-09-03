FROM python:3.11-slim

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and data
COPY . .

EXPOSE 10000

# Start FastAPI (Render will route to this port)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
