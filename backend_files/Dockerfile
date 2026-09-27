
# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the 'best_model' directory into the container
# This ensures our saved model is available
COPY best_model /app/best_model

# Copy the application files into the container at /app
COPY app.py .

# Expose the port that the application will run on
EXPOSE 8000

# Run the application using Gunicorn
# Gunicorn is a WSGI HTTP server for Unix, often used for deploying Python web applications.
# We'll run it on port 8000, binding to all network interfaces (0.0.0.0)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
