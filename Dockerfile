FROM python:3.9.19-slim-bullseye

WORKDIR /home/rl_research

COPY . .

RUN pip install -r requirements.txt

CMD python -u main.py
