FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir sqlalchemy python-jose[cryptography] bcrypt openpyxl shap torch --index-url https://download.pytorch.org/whl/cpu

# Copy application files
COPY . .

# Run the pipeline if data isn't pre-processed
RUN python main.py

EXPOSE 8000 8501

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
