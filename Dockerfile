FROM python:3.10-slim

WORKDIR /

# Install dependencies
COPY requirements.txt /requirements.txt

RUN uv pip install --upgrade -r /requirements.txt --no-cache-dir --system

# Add files
COPY handler.py /

# Run the handler
CMD python -u /handler.py
