FROM python:3.9.19-slim-bullseye

WORKDIR /home/rl_research

COPY . .

RUN apt-get update && apt-get install -y build-essential

RUN pip install -r requirements2.txt
RUN pip install -r requirements.txt 

CMD python -u main.py
