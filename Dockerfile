# ================================
# Base Image
# ================================
FROM python:3.10-slim

# ================================
# Environment Variables
# ================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ================================
# Working Directory
# ================================
WORKDIR /app

# ================================
# Copy requirements
# ================================
COPY requirements.txt .

# ================================
# Install dependencies
# ================================
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# ================================
# Copy project
# ================================
COPY . .

# ================================
# Create upload folder
# ================================
RUN mkdir -p static/uploads

# ================================
# Expose Flask Port
# ================================
EXPOSE 5000

# ================================
# Start Application
# ================================
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]