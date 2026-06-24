FROM python:3.11-slim

# System deps (build tools for some wheels, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root user (Hugging Face Spaces convention: uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python dependencies first (better layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY --chown=user . .

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860

# XSRF/CORS disabled so file uploads work behind the Hugging Face proxy/iframe
# (otherwise the uploader returns "AxiosError: 403"). Standard for HF Spaces.
CMD ["streamlit", "run", "version2/demo.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", \
     "--server.enableCORS=false"]
