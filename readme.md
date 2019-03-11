
# Spacy & NLTK. 

A script to provide summary stats associated with a text file of your choosing!
Entity label and types will be written to csv with frequency and sentiment scores.

### Installing

Install everything within the requirements.txt

```
pip install -r ./requirements.txt
```


## Deployment

### entity_sentiment.py

Run this script and pass the location of a text file. 

It'll output the top 5 entity labels and types by frequency, along with the average sentiment scores for the sentences those entity exist within. 

This will be written to csv within the data folder.

```
python3 entity_sentiment.py text_file.txt
```

### Next Steps
Expand on inputs options and output summary stats.
Allow for updating/retraining model with new entities & labels.

## Acknowledgments/Sources

* Video on Spacy NER: SPACY'S ENTITY RECOGNITION MODEL: incremental parsing with Bloom embeddings & residual CNNs. Source: https://www.youtube.com/watch?v=sqDHBH9IjRU
* Original training data for Spacy: https://catalog.ldc.upenn.edu/LDC2013T19
* Textblob for Sentiment: https://textblob.readthedocs.io/en/dev/
* Spacy model of choice: https://github.com/explosion/spacy-models/releases/tag/en_core_web_md-2.0.0

