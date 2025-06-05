FROM tiangolo/uvicorn-gunicorn:python3.10

WORKDIR /app/backend

COPY ./backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./backend /app/backend

CMD [ "python", "main.py" ]

EXPOSE 8000

