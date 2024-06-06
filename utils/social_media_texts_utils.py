import preprocessor as p 
import json
import re
import wordninja
import re
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Data Cleaning
def split_hash_tag(strings):
    
    p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.RESERVED)
    clean_data = p.clean(strings)  # using lib to clean URL, emoji...
    clean_data = clean_data.split(' ')
    
    for i in range(len(clean_data)):
        if clean_data[i].startswith("#") or clean_data[i].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i]) # split compound hashtags
        else:
            clean_data[i] = [clean_data[i]]
    clean_data = [j for i in clean_data for j in i]

    return ' '.join(clean_data)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def clean_text(text):
    """Function to clean text using RegEx operations, removal of stopwords, and lemmatization."""
    text_lst = text.split(' ')
    text_lst = [token.lower() for token in text_lst]
    text_lst = [token for token in text_lst if token not in stop_words]
    text_lst = [re.sub(r'[^\w\s]', '', token, re.UNICODE) for token in text_lst]
    text_lst = [token for token in text_lst if token not in stop_words]
    text_lst = [lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1])) for tag in pos_tag(text_lst) if get_wordnet_pos(tag[1]) is not None]
    text_lst = [token for token in text_lst if token not in stop_words]
    text = ' '.join(text_lst)
    text = text.lstrip().rstrip()
    return text
