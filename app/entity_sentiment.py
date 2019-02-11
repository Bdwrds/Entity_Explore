
"""
Script to provide a summary of entities labels/types and 
their sentiments at aggregate.
"""
import spacy
import nltk
import sys
import pandas as pd
from textblob import TextBlob


def get_entity_frame(corpus):
	named_entities = []
	sentences = nltk.sent_tokenize(corpus)
	for sentence in sentences:	
	    temp_entity_name = ''
	    rw = 1
	    temp_named_entity = None
	    sentence = nlp(sentence)
	    for word in sentence:
	        term = word.text 
	        tag = word.ent_type_
	        if tag:
	            temp_entity_name = ' '.join([temp_entity_name, term]).strip()
	            temp_named_entity = (temp_entity_name, tag, rw)
	        else:
	            if temp_named_entity:
	                named_entities.append(temp_named_entity)
	                temp_entity_name = ''
	                temp_named_entity = None
	                rw += 1

	entity_frame = pd.DataFrame(named_entities, columns=['Entity Name', 'Entity Type', 'sentence'])
	return(entity_frame)


def get_setence_sentiment(corpus):
	sentence_sentiment = []
	sentences = nltk.sent_tokenize(corpus)
	rw = 1
	for sentence in sentences:
	    sentence_blob = TextBlob(sentence)
	    sentence_pol = sentence_blob.sentiment.polarity
	    sentence_sentiment.append([rw, sentence_pol])
	sentiment_frame = pd.DataFrame(sentence_sentiment, columns=['sentence','Sentiment'])
	return(sentiment_frame)



def get_top_x_entities(top_entities, num=5):
	top_entities = (entity_frame.groupby(by=['Entity Name', 'Entity Type'])
	                           .agg({'Entity Type':'size','Sentiment':'mean'})
	                           .rename(columns={'Entity Type':'Frequency','Sentiment':'Mean_Sentiment'})
	                           .sort_values(by='Frequency',ascending=False)
	                           .reset_index())
	return(top_entities.T.iloc[:,:num])


def get_top_x_entities_types(top_entities, num=5):
	top_entities = (entity_frame.groupby(by=['Entity Type'])
	                           .agg({'Entity Type':'size','Sentiment':'mean'})
	                           .rename(columns={'Entity Type':'Frequency','Sentiment':'Mean_Sentiment'})
	                           .sort_values(by='Frequency',ascending=False)
	                           .reset_index())
	return(top_entities.T.iloc[:,:num])


if __name__ == "__main__":

    print(("Running script to get Entity level sentiment!"))
    nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
    
    with open(sys.argv[1], 'r') as f:
    	text = f.read()
    f.close()

    entity_frame = get_entity_frame(text)
    sentiment_frame = get_setence_sentiment(text)
    entity_frame = entity_frame.join(sentiment_frame[['Sentiment']], on='sentence', lsuffix='_EN')

    print(get_top_x_entities(entity_frame, 5))
    print(get_top_x_entities_types(entity_frame, 5))

    entity_frame.to_csv("data/entity_frame.csv",sep=",", index=False)


