FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app


COPY [ "Pipfile",  "Pipfile.lock", "./"]
COPY [ "starter.py", "./" ]

ADD requirements.txt .
RUN pip install -r requirements.txt

CMD [ "python", "./starter.py" , "2021", "4"]
