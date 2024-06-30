# Use the official Python base image
FROM python:3.10.14-bookworm

# Upgrade pip and install dependencies
RUN pip install --upgrade pip

# Copy the application source code
COPY src /app/src

# Copy the entrypoint script to the container
COPY entrypoint.sh /app/entrypoint.sh

# Set the working directory
WORKDIR /app

# Create the trained_models directory
RUN mkdir -p /app/src/trained_models

# Adjust permissions (if needed)
RUN chmod -R 777 /app/src

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Install the Python dependencies
RUN pip install -r /app/src/requirements.txt

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=${PYTHONPATH}:/app/src

# Set the entrypoint to the script
ENTRYPOINT ["/app/entrypoint.sh"]