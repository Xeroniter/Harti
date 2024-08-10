FROM python:3.9-slim

WORKDIR /application

COPY . /application

RUN pip install --upgrade pip

RUN pip install -r /application/requirements.txt

EXPOSE 8080

CMD ["fastapi", "run", "/application/app.py", "--host", "0.0.0.0", "--port", "8080"]