FROM python:3.6

#set wkdir
WORKDIR /app

COPY requirements.txt /app
RUN pip install  -r ./requirements.txt

COPY app /app

RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader punkt

CMD ["python", "entity_sentiment.py"]

