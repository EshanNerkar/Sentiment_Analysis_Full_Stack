# Use an official Python runtime as a parent image
FROM python:3.8

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Download NLTK data without user interaction
RUN python -m "import nltk; nltk.download('stopwords', quiet=True)"
RUN python -m "import nltk; nltk.download('punkt', quiet=True)"

# Copy the rest of the application code into the container
COPY . /app/

# Expose port 8000
EXPOSE 8000

# Start the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
