import pandas as pd
import fasttext
import json
import re
import nltk
import psycopg2
from ast import literal_eval
from config import HOST,DATABASE,USER,PASSWORD,PORT,TABLE,CLEAN
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()
from autocorrect import Speller
spell = Speller()
pd.set_option('display.max_colwidth', None)

model = fasttext.load_model('lid.176.ftz') # file not provided

def detect_lang(text):
    lang = model.predict(text, k=1)
    lang = lang[0][0][-2:]
    return lang

# custom built islamic dicionary
with open('islamic_dict.json') as json_file:
    islamic_dict = json.load(json_file)
    
import_num = islamic_dict['number']
anti_cleaner = islamic_dict['immune']
islamic_replace = islamic_dict['replacement']
topic_keywords = islamic_dict['topics']
issues_keywords = islamic_dict['issues']
stopword_addition = islamic_dict['stopwords']

stopword = nltk.corpus.stopwords.words('english')
stopword = [c.replace("'","") for c in stopword]
stopword.extend(stopword_addition)

def cleaner_readable(text):
    text = text.lower()
    # protect key numbers, change " 99 " to "nintynine"
    if list(import_num.keys())[0] in text:
        text = text.replace(list(import_num.keys())[0],list(import_num.values())[0])
    # get words with alphabets
    text = ' '.join(re.findall('[a-zA-Z]+',text))
    return text

def cleaner_ss(text):
    # grant immunity to important words
    result = []
    text = word_tokenize(text)
    for t in text:
        if t in anti_cleaner:
            result.append(t)
        else:
            # stemming and spelling correction
            if len(t)>3:
                t = ps.stem(t)
                t = spell(t)
                # remove stopwords
                if t not in stopword:
                    result.append(t)
    return ' '.join(result)


def clean_cache():
    # input
    conn = psycopg2.connect(host = HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)
    sql = f'''
    select {TABLE}."ReviewID", {TABLE}."translated"
    from {TABLE}
    left join {CLEAN}
    on {TABLE}."ReviewID" = {CLEAN}."ReviewID"
    where {CLEAN}."ReviewID" is null
    '''
    df = pd.read_sql_query(sql,con=conn,index_col='ReviewID')
    cur = conn.cursor()

    count_clean = [0,0]
    for id, text in df.iterrows():
        text = text[0]
        if detect_lang(text) == 'en':
            text_readable = cleaner_readable(text)
            text_cleaned = cleaner_ss(text_readable)
            cur.execute(f'''INSERT into {CLEAN} ("ReviewID",review_readable,review_cleaned) VALUES ({id},'{text_readable}','{text_cleaned}');''')
            conn.commit()
            count_clean[0]+=1
        else:
            count_clean[1]+=1

    print(f'Added {count_clean[0]} into {CLEAN}, Skipped {count_clean[1]} records')
    # Get number of records in Clean
    cur.execute(f"select count(*) from {CLEAN}")
    cleaned_records = cur.fetchone()
    print(f"Number of records in {CLEAN}: ",cleaned_records[0])

    # Get number of records in Reviews
    cur.execute(f"select count(*) from {TABLE}")
    records = cur.fetchone()
    print(f"Number of records in {TABLE}: ",records[0])
    
    cur.close()

    return (cleaned_records[0],records[0])

def clean_escape(text):
    text = repr(text)
    text = text.replace(r'\\n',' ',99)
    text = text.replace(r'\\ n',' ',99)
    text = text.replace(r'\n',' ',99)
    text = text.replace(r'\"',' ',99)
    text = text.replace(r'  ',' ',99)
    text = literal_eval(text)
    return text
