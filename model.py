import json
import pandas as pd
import numpy as np
from datetime import datetime
from clean import clean_escape


def read_output(select_date,select_os='app'):
    if select_os == 'app':
        with open(f"model_output/app_{select_date}.json") as f:
            return json.loads(f.read())
    else:
        filename = select_os + '_' + select_date + '.json'
        try:
            with open("model_output/"+filename) as f:
                return json.loads(f.read())
        except:
            with open("model_output/sample_output.json") as f:
                return json.loads(f.read())


def run_model(df):
    # create output variable
    model_output = {}

    # remove duplicate
    df = df.drop_duplicates(subset = ['translated'])
    date_start = df['Start Date'].min()
    date_end = df['Start Date'].max()
    model_output['as_of_date'] = date_start.strftime("%d-%m-%Y")+' - ' +date_end.strftime("%d-%m-%Y")
    print('Added as_of_date to model_output')

    # # Charts
    track_duration = datetime.now()     # for duration tracking

    df['Country/Region/Language'] = df['Country/Region/Language'].str.upper()
    country = df[['Platform','Country/Region/Language','Review']].groupby(['Platform','Country/Region/Language'])['Review'].count().unstack('Platform',fill_value=0)
    country['total'] = country.sum(axis=1)
    chart_country = country.sort_values('total',ascending=False)
    chart_country = chart_country.head(10)
    chart_country = chart_country.to_dict()
    model_output['chart_country'] = chart_country
    print(f'Added chart_country to model_output: {datetime.now()-track_duration}')        # for duration tracking

    track_duration = datetime.now()     # for duration tracking

    chart_piebyrating = df.Rating.value_counts()
    chart_piebyrating = pd.DataFrame(chart_piebyrating.sort_index())
    chart_piebyrating = chart_piebyrating.to_dict()
    model_output['chart_pie'] = chart_piebyrating
    print(f'Added chart_pie to model_output: {datetime.now()-track_duration}')        # for duration tracking

    track_duration = datetime.now()     # for duration tracking

    chart_ratingbydate = df[['Rating','Start Date']].groupby(['Start Date','Rating'])['Rating'].count().unstack('Rating')
    chart_ratingbydate.fillna(0,inplace=True)
    chart_ratingbydate.index = [str(k)[:10] for k in chart_ratingbydate.index.to_list()]
    chart_ratingbydate = chart_ratingbydate.to_dict()
    model_output['chart_ratingbydate'] = chart_ratingbydate
    print(f'Added chart_ratingbydate to model_output: {datetime.now()-track_duration}')        # for duration tracking

    # # # Remove non-English text

    df_en = df.copy()

    import fasttext

    model = fasttext.load_model('data/lid.176.ftz')

    def detect_lang(text):
        lang = model.predict(text, k=1)
        lang = lang[0][0][-2:]
        return lang

    track_duration = datetime.now()     # for duration tracking

    print('Detecting language ...')
    df_en['detected_language'] = df_en.translated.apply(detect_lang)
    df_en = df_en[df_en['detected_language'] == 'en']
    df_en.drop(columns=['detected_language'],inplace=True)
    print(f'Reduced rows from {len(df)} to {len(df_en)}')
    print(f'Removed non-English reviews took : {datetime.now()-track_duration}')        # for duration tracking

    # # Define Important Words
    # for immunity against cleaning

    # custom built islamic dicionary
    with open('data/islamic_dict.json') as json_file:
        islamic_dict = json.load(json_file)
        
    import_num = islamic_dict['number']
    anti_cleaner = islamic_dict['immune']
    islamic_replace = islamic_dict['replacement']
    topic_keywords = islamic_dict['topics']
    issues_keywords = islamic_dict['issues']
    stopword_addition = islamic_dict['stopwords']


    # # # Clean Data

    # import string
    # import re
    # import nltk
    # # nltk.download('stopwords')
    # from nltk.tokenize import word_tokenize
    # from nltk.stem import PorterStemmer
    # # nltk.download('punkt')
    # ps = PorterStemmer()
    # from autocorrect import Speller
    # spell = Speller()

    # track_duration = datetime.now()     # for duration tracking

    # stopword = nltk.corpus.stopwords.words('english')
    # stopword = [c.replace("'","") for c in stopword]
    # stopword.extend(stopword_addition)

    # def cleaner_readable(text):
    #     text = text.lower()
    #     # protect key numbers, change " 99 " to "nintynine"
    #     if list(import_num.keys())[0] in text:
    #         text = text.replace(list(import_num.keys())[0],list(import_num.values())[0])
    #     # get words with alphabets
    #     text = ' '.join(re.findall('[a-zA-Z]+',text))
    #     return text
        
    # import multiprocessing
    # import dask.dataframe as dd

    # def cleaner_ss(text):
    #     # grant immunity to important words
    #     result = []
    #     text = word_tokenize(text)
    #     for t in text:
    #         if t in anti_cleaner:
    #             result.append(t)
    #         else:
    #             # stemming and spelling correction
    #             if len(t)>3:
    #                 t = ps.stem(t)
    #                 t = spell(t)
    #                 # remove stopwords
    #                 if t not in stopword:
    #                     result.append(t)
    #     return ' '.join(result)

    # n_core = multiprocessing.cpu_count()
    # print(f'cleaning text data using {n_core} cores')
    # df_en['review_readable'] = df_en['translated'].apply(cleaner_readable)
    
    # ddata = dd.from_pandas(df_en,npartitions=n_core*2)
    # df_en['review_cleaned'] = ddata.map_partitions(lambda df_en: df_en.apply((lambda row: cleaner_ss(row['review_readable'])),axis=1)).compute(scheduler='processes')
    # print(f'cleaned text data and stop words removed: {datetime.now()-track_duration}')        # for duration tracking


    # # # Replace Islamic words

    # def replace_islamic(text):
    #     result = []
    #     for t in text.split(' '):
    #         try:
    #             result.append(islamic_replace[t].lower())
    #         except KeyError:
    #             result.append(t)
    #     return ' '.join(result)

    # track_duration = datetime.now()     # for duration tracking

    # df_en['review_cleaned'] = df_en['review_cleaned'].apply(replace_islamic)

    # # filter by length > 10
    # df_en = df_en[df_en.review_cleaned.str.len()>10]
    # # filter by number of words > 3
    # df_en = df_en[df_en.review_cleaned.str.split().apply(len) > 3]
    # print(f'Replace islamic words took : {datetime.now()-track_duration}')        # for duration tracking


    # # Polarity

    from textblob import TextBlob

    track_duration = datetime.now()     # for duration tracking

    df_en_hist = df_en.copy()
    df_en_hist_columns = list(df_en_hist.columns)
    df_en_hist['polarity'] = df_en_hist.review_readable.apply(lambda x : TextBlob(x).sentiment.polarity)
    df_en_hist.drop(columns=df_en_hist_columns,inplace=True)
    bins = [i/100 for i in range(-100,101)]
    df_en_hist['bin'] = pd.cut(df_en_hist['polarity'],bins)
    chart_histpolarity = df_en_hist.groupby(['bin']).count()
    chart_histpolarity.index = [i.right for i in chart_histpolarity.index.to_list()]
    chart_histpolarity = chart_histpolarity.to_dict()
    model_output['chart_histpolarity'] = chart_histpolarity
    print(f'Added chart_histpolarity to model_output: {datetime.now()-track_duration}')        # for duration tracking


    # # POS Tagging, Nouns & Adjustives

    from collections import Counter
    import nltk
    import re
    from nltk.tokenize import word_tokenize

    def speech_tagging(text):
        text = re.split('\W+',text.lower())
        text = list(filter(None,text))
        text = nltk.pos_tag(text)
        return text

    def nouns(text):
        is_noun = lambda pos:pos[:2] == 'NN'
        return [word for (word,pos) in text if is_noun(pos)]

    def adjective(text):
        is_adjective = lambda pos: pos[:2] == 'JJ'
        return [word for (word,pos) in text if is_adjective(pos)]

    def top_10_words(text):
        words = [word for i in range(len(text)) for word in text[i]]
        counts = Counter(words)
        return [word for (word,count) in counts.most_common(10)]

    track_duration = datetime.now()     # for duration tracking

    df_en_pos = df_en.copy()
    df_en_pos.reset_index(drop=True,inplace=True)
    df_en_pos['review_tagged'] = df_en_pos['review_cleaned'].apply(speech_tagging)
    df_en_pos['nouns'] = df_en_pos['review_tagged'].apply(nouns)
    df_en_pos['adjectives'] = df_en_pos['review_tagged'].apply(adjective)

    print("Top-10 most frequent nouns are: ")
    print(top_10_words(df_en_pos['nouns'].tolist()))

    print("Top-10 most frequent adjectives are: ")
    print(top_10_words(df_en_pos['adjectives'].tolist()))

    # # Indicative Adjectives

    from collections import defaultdict

    def counter(wordlist):
        words = [word for i in range(len(wordlist)) for word in wordlist[i]]
        counts = Counter(words)
        return len(words), counts
    
    def top_10_indicative_adjectives(ratingAdjDict, ratingAdjCount, adjDict, adjCount):
        relativeEntropy = defaultdict(int)
        top_10_indicative_adjectives = []
        for adj in ratingAdjDict:
            probWordInRating = ratingAdjDict[adj]/ratingAdjCount
            probWordInAll = adjDict[adj]/adjCount
            relativeEntropy[adj] = probWordInRating * np.log(probWordInRating/probWordInAll)
        top_10_List = sorted(relativeEntropy.items(), key=lambda x: x[1], reverse=True)[:10]
        for adj, relativeEntropy in top_10_List:
            top_10_indicative_adjectives.append(adj)
        return top_10_indicative_adjectives

    ratings = sorted(df_en_pos['Rating'].unique())
    adjCount, adjDict = counter(df_en_pos['adjectives'].tolist())

    for rating in ratings:
        ratingAdjCount, ratingAdjDict = counter(df_en_pos[df_en_pos['Rating']==rating]['adjectives'].tolist())
        print('Top 10 indicative adjectives for rating ', rating, ': ')
        print(top_10_indicative_adjectives(ratingAdjDict, ratingAdjCount, adjDict, adjCount))

    print(f'Get indicative adjectives took : {datetime.now()-track_duration}')        # for duration tracking

    from gensim.models import TfidfModel
    from gensim.corpora import Dictionary

    track_duration = datetime.now()     # for duration tracking

    df_en_pos['review_token'] = df_en_pos.review_cleaned.apply(lambda x: word_tokenize(x))

    dct = Dictionary(df_en_pos.review_token.tolist())
    corpus = [dct.doc2bow(line) for line in df_en_pos.review_token]
    model = TfidfModel(corpus)


    # # Word Cloud
    # on TFIDF keywords

    df_en_vec = pd.concat([df_en_pos[['Platform','Rating','translated','review_readable','review_cleaned']],pd.Series(corpus)],axis=1)
    df_en_vec.columns = ['Platform','Rating','translated','review_readable','review_cleaned','term_freq']

    def apply_tfidf(tf):
        keywords = []
        for i,v in model[tf]:
            if v>0.2:
                keywords.append(dct[i])
        return ' '.join(keywords)

    df_en_vec['tfidf'] = df_en_vec.term_freq.apply(apply_tfidf)
    df_en_vec.drop(columns = ['term_freq'],inplace=True)
    wordcloud = {}
    def apply_wordcloud(text):
        for t in text.split(' '):
            if t in wordcloud:
                wordcloud[t] += 1
            else:
                wordcloud[t] = 1

    df_en_vec['tfidf'].apply(apply_wordcloud)
    wordcloud = pd.DataFrame.from_dict(wordcloud,orient='index',columns=['count'])
    wordcloud = wordcloud.sort_values('count',ascending=False).head(100)
    wordcloud = wordcloud.to_dict()['count']
    # normalising wordcloud display
    wordcloud_max = max(wordcloud.values())
    wordcloud_min = min(wordcloud.values())
    for k in wordcloud:
        wordcloud[k] = (wordcloud[k]-wordcloud_min)/(wordcloud_max-wordcloud_min)*(50)+10
    # print(wordcloud)
    model_output['wordcloud'] = wordcloud
    print(f'Get wordcloud using TFIDF took: {datetime.now()-track_duration}')        # for duration tracking


    # # Get Topics

    def get_topic(text):
        topic_list = list(topic_keywords.keys())
        result = []
        for topic in topic_list:
            count = 0
            for word in list(set(text.split(' '))):
                if word in topic_keywords[topic]:
                    count+=1
            result.append(count)
        if max(result) == 0:
            return 'others'
        else:
            return topic_list[result.index(max(result))]

    track_duration = datetime.now()     # for duration tracking

    df_en_vec['topic'] = df_en_vec.review_cleaned.apply(get_topic)

    plot_label = {}
    for i in df_en_vec.topic.value_counts().index:
        plot_label[i] = i+' ('+str(df_en_vec.topic.value_counts()[i])+')'

    chart_topics = df_en_vec.groupby('topic')['Rating'].value_counts(normalize=True)*100
    chart_topics = pd.Series(chart_topics.rename(index = plot_label))
    plot_label = list(plot_label.values())
    plot_label.sort()

    chart_topics_dict = {'label':plot_label,'1':[],'2':[],'3':[],'4':[],'5':[]}

    # # fill missing values
    for k in plot_label:
        for i in range(1,6):
            try:
                chart_topics_dict[str(i)].append(chart_topics[k][i])
            except KeyError:
                chart_topics_dict[str(i)].append(0.0)
    model_output['chart_topics'] = chart_topics_dict
    print(f'Get chart_topics took : {datetime.now()-track_duration}')        # for duration tracking


    # # Get Issues

    def get_issues(text):
        issues_list = list(issues_keywords.keys())
        result = []
        for issues in issues_list:
            count = 0
            for word in list(set(text.split(' '))):
                if word in issues_keywords[issues]:
                    count+=1
            result.append(count)
        if max(result) == 0:
            return 'others'
        else:
            return issues_list[result.index(max(result))]

    track_duration = datetime.now()     # for duration tracking
        
    df_en_issues = df_en_vec.copy()
    df_en_issues['issues'] = df_en_issues.review_cleaned.apply(get_issues)

    from gensim.summarization.summarizer import summarize

    paragraph_issues = {}
    list_issues = list(df_en_issues.issues.unique())
    list_issues.sort()
    for issues in list_issues:
        df_en_issues_i = df_en_issues[df_en_issues['issues']==issues]
        paragraph_issues[issues.upper()]=clean_escape(summarize('. '.join(df_en_issues_i.review_readable.tolist()),word_count = 50))

    model_output['paragraph_issues'] = paragraph_issues
    print(f'Get paragraph_issues took : {datetime.now()-track_duration}')        # for duration tracking

    # # Page ranking

    # extract word vectors

    track_duration = datetime.now()     # for duration tracking

    word_embeddings = {}
    f = open('data/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    print('begin pagerank algorithm')
    clean_sentences = df_en_vec.review_cleaned.tolist()
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    import torch
    import torch.nn as nn

    print('calculating cosine similarity ...')
    progresscount = 0

    sim_mat = np.zeros([len(clean_sentences), len(clean_sentences)])
    cos = nn.CosineSimilarity(dim=0,eps=1e-6)

    for i in range(len(clean_sentences)):
        for j in range(len(clean_sentences)):
                if i != j:
                    sim_mat[i][j] = float(cos(torch.from_numpy(sentence_vectors[i]), torch.from_numpy(sentence_vectors[j])))
        progresscount += 1
        print(f'{progresscount}/{len(clean_sentences)}',end='\r')
    print(f'Get cosine similarity took : {datetime.now()-track_duration}')        # for duration tracking

    import networkx as nx
    
    track_duration = datetime.now()     # for duration tracking

    print('graphing features ')
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    s_scores = pd.Series(scores)
    s_scores.name = 'pagerank_score'
    df_paragraph_features = pd.concat([df_en_vec,s_scores],axis=1).sort_values(by=['pagerank_score'],ascending=False)
    df_paragraph_features = df_paragraph_features.groupby('topic').head(3).reset_index()
    df_paragraph_features = df_paragraph_features[['topic','Rating','translated']].head(10)
    df_paragraph_features['translated'] = df_paragraph_features['translated'].apply(clean_escape)
    df_paragraph_features.to_dict('index')
    model_output['paragraph_features'] = df_paragraph_features.to_dict('index')
    print(f'Get paragraph_features took : {datetime.now()-track_duration}')        # for duration tracking

    return model_output


if __name__ == "__main__":
    print(run_model())


