FROM python:3.12-slim AS builder

WORKDIR /app

# Install Python dependencies into a separate prefix
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-prod.txt

# --- Production stage ---
FROM python:3.12-slim

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
