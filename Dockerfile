FROM python-3.9

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# copy all content from local to docker container 
COPY . /app/

# Install requirements including dlib
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Run FastAPI server with the specified command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
