#!/usr/bin/env python
# coding: utf-8

# ## Complete Text Processing 

# ### General Feature Extraction
# - File loading
# - Word counts
# - Characters count
# - Average characters per word
# - Stop words count
# - Count #HashTags and @Mentions
# - If numeric digits are present in twitts
# - Upper case word counts

# ### Preprocessing and Cleaning
# - Lower case
# - Contraction to Expansion
# - Emails removal and counts
# - URLs removal and counts
# - Removal of RT
# - Removal of Special Characters
# - Removal of multiple spaces
# - Removal of HTML tags
# - Removal of accented characters
# - Removal of Stop Words
# - Conversion into base form of words
# - Common Occuring words Removal
# - Rare Occuring words Removal
# - Word Cloud
# - Spelling Correction
# - Tokenization
# - Lemmatization
# - Detecting Entities using NER
# - Noun Detection
# - Language Detection
# - Sentence Translation
# - Using Inbuilt Sentiment Classifier

# In[1]:


import pandas as pd
import numpy as np
import spacy


# In[2]:


from spacy.lang.en.stop_words import STOP_WORDS as stopwords


# In[ ]:


df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitter4000.csv', encoding = 'latin1')


# In[ ]:


df


# In[ ]:


df['sentiment'].value_counts()


# ## Word Counts

# In[ ]:


len('this is text'.split())


# In[ ]:


df['word_counts'] = df['twitts'].apply(lambda x: len(str(x).split()))


# In[ ]:


df.sample(5)


# In[ ]:


df['word_counts'].max()


# In[ ]:


df['word_counts'].min()


# In[ ]:


df[df['word_counts']==1]


# # Characters Count

# In[ ]:


len('this is')


# In[ ]:


def char_counts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)


# In[ ]:


char_counts('this is')


# In[ ]:


df['char_counts'] = df['twitts'].apply(lambda x: char_counts(str(x)))


# In[ ]:


df.sample(5)


# ## Average Word Length

# In[ ]:


x = 'this is' # 6/2 = 3
y = 'thankyou guys' # 12/2 = 6


# In[ ]:


df['avg_word_len'] = df['char_counts']/df['word_counts']


# In[ ]:


df.sample(4)


# ## Stop Words Count 

# In[ ]:


print(stopwords)


# In[ ]:


len(stopwords)


# In[ ]:


x = 'this is the text data'


# In[ ]:


x.split()


# In[ ]:


[t for t in x.split() if t in stopwords]


# In[ ]:


len([t for t in x.split() if t in stopwords])


# In[ ]:


df['stop_words_len'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t in stopwords]))


# In[ ]:


df.sample(5)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Count #HashTags and @Mentions 

# In[ ]:


x = 'this is #hashtag and this is @mention'


# In[ ]:


x.split()


# In[ ]:


[t for t in x.split() if t.startswith('@')]


# In[ ]:


len([t for t in x.split() if t.startswith('@')])


# In[ ]:


df['hashtags_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))


# In[ ]:


df['mentions_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))


# In[ ]:


df.sample(5)


# In[ ]:





# ## If numeric digits are present in twitts

# In[ ]:


x = 'this is 1 and 2'


# In[ ]:


x.split()


# In[ ]:


x.split()[3].isdigit()


# In[ ]:


[t for t in x.split() if t.isdigit()]


# In[ ]:


df['numerics_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))


# In[ ]:


df.sample(5)


# ## UPPER case words count 

# In[ ]:


x = 'I AM HAPPY'
y = 'i am happy'


# In[ ]:


[t for t in x.split() if t.isupper()]


# In[ ]:


df['upper_counts'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isupper()]))


# In[ ]:


df.sample(5)


# In[ ]:


df.iloc[3962]['twitts']


# In[ ]:





# In[ ]:





# # Preprocessing and Cleaning

# ## Lower Case Conversion 

# In[ ]:


x = 'this is Text'


# In[ ]:


x.lower()


# In[ ]:


x = 45.0
str(x).lower()


# In[ ]:


df['twitts'] = df['twitts'].apply(lambda x: str(x).lower())


# In[ ]:


df.sample(5)


# ## Contraction to Expansion 

# In[ ]:


contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
" u ": " you ",
" ur ": " your ",
" n ": " and ",
"won't": "would not",
'dis': 'this',
'bak': 'back',
'brng': 'bring'}


# In[ ]:


x = "i'm don't he'll" # "i am do not he will"


# In[ ]:


def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x
    


# In[ ]:


cont_to_exp(x)


# In[ ]:


get_ipython().run_cell_magic('timeit', '', "df['twitts'] = df['twitts'].apply(lambda x: cont_to_exp(x))")


# In[ ]:


df.sample(5)


# In[ ]:





# ## Count and Remove Emails 

# In[ ]:


import re


# In[ ]:


df[df['twitts'].str.contains('hotmail.com')]


# In[ ]:


df.iloc[3713]['twitts']


# In[ ]:


x = '@securerecs arghh me please  markbradbury_16@hotmail.com'


# In[ ]:


re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', x)


# In[ ]:


df['emails'] = df['twitts'].apply(lambda x: re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x))


# In[ ]:


df['emails_count'] = df['emails'].apply(lambda x: len(x))


# In[ ]:


df[df['emails_count']>0]


# In[ ]:


re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)


# In[ ]:


df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x))


# In[ ]:


df[df['emails_count']>0]


# In[ ]:





# ## Count URLs and Remove it 

# In[ ]:


x = 'hi, thanks to watching it. for more visit https://youtube.com/kgptalkie'


# In[ ]:


#shh://git@git.com:username/repo.git=riif?%


# In[ ]:


re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)


# In[ ]:


df['url_flags'] = df['twitts'].apply(lambda x: len(re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))


# In[ ]:


df[df['url_flags']>0].sample(5)


# In[ ]:


x


# In[ ]:


re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)


# In[ ]:


df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x))


# In[ ]:


df.sample(5)


# In[ ]:





# ## Remove RT 

# In[ ]:


df[df['twitts'].str.contains('rt')]


# In[ ]:


x = 'rt @username: hello hirt'


# In[ ]:


re.sub(r'\brt\b', '', x).strip()


# In[ ]:


df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'\brt\b', '', x).strip())


# In[ ]:





# In[ ]:





# In[ ]:





# ## Special Chars removal or punctuation removal 

# In[ ]:


df.sample(3)


# In[ ]:


x = '@duyku apparently i was not ready enough... i...'


# In[ ]:


re.sub(r'[^\w ]+', "", x)


# In[ ]:


df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'[^\w ]+', "", x))


# In[ ]:


df.sample(5)


# In[ ]:





# ## Remove multiple spaces `"hi   hello    "`

# In[ ]:


x =  'hi    hello     how are you'


# In[ ]:


' '.join(x.split())


# In[ ]:


df['twitts'] = df['twitts'].apply(lambda x: ' '.join(x.split()))


# In[ ]:





# ## Remove HTML tags

# In[ ]:


get_ipython().system('pip install beautifulsoup4')


# In[ ]:


from bs4 import BeautifulSoup


# In[ ]:


x = '<html><h1> thanks for watching it </h1></html>'


# In[ ]:


x.replace('<html><h1>', '').replace('</h1></html>', '') #not rec


# In[ ]:


BeautifulSoup(x, 'lxml').get_text().strip()


# In[ ]:


get_ipython().run_cell_magic('time', '', "df['twitts'] = df['twitts'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text().strip())")


# In[ ]:





# ## Remove Accented Chars 

# In[ ]:


x = 'Áccěntěd těxt'


# In[ ]:


import unicodedata


# In[ ]:


def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x


# In[ ]:


remove_accented_chars(x)


# In[ ]:


df['twitts'] = df['twitts'].apply(lambda x: remove_accented_chars(x))


# In[ ]:





# In[ ]:





# In[ ]:





# ## Remove Stop Words 

# In[ ]:


x = 'this is a stop words'


# In[ ]:


' '.join([t for t in x.split() if t not in stopwords])


# In[ ]:


df['twitts_no_stop'] = df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in stopwords]))


# In[ ]:


df.sample(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Convert into base or root form of word 

# In[ ]:


nlp = spacy.load('en_core_web_sm')


# In[ ]:


x = 'this is chocolates. what is times? this balls'


# In[ ]:


def make_to_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)
    
    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)


# In[ ]:


make_to_base(x)


# In[ ]:


df['twitts'] = df['twitts'].apply(lambda x: make_to_base(x))


# In[ ]:


df.sample(5)


# In[ ]:





# ## Common words removal 

# In[ ]:


x = 'this is this okay bye'


# In[ ]:


text = ' '.join(df['twitts'])


# In[ ]:


len(text)


# In[ ]:


text = text.split()


# In[ ]:


len(text)


# In[ ]:


freq_comm = pd.Series(text).value_counts()


# In[ ]:


f20 = freq_comm[:20]


# In[ ]:


f20


# In[ ]:


df['twitts'] = df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in f20]))


# In[ ]:


df.sample(5)


# In[ ]:





# ## Rare words removal 

# In[ ]:


rare20 = freq_comm.tail(20)


# In[ ]:


df['twitts'] = df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in rare20]))


# In[ ]:


df.sample(5)


# ## Word Cloud Visualization 

# In[ ]:


# !pip install wordcloud


# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


text = ' '.join(df['twitts'])


# In[ ]:


len(text)


# In[ ]:


wc = WordCloud(width=800, height=400).generate(text)
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[ ]:





# ## Spelling Correction 

# In[ ]:


get_ipython().system('pip install -U textblob')


# In[ ]:


get_ipython().system('python -m textblob.download_corpora')


# In[ ]:


from textblob import TextBlob


# In[ ]:


x = 'thankks forr waching it'


# In[ ]:


x = TextBlob(x).correct()


# In[ ]:


x


# ## Tokenization using TextBlob
# 

# In[ ]:


x = 'thanks#watching this video. please like it'


# In[ ]:


TextBlob(x).words


# In[ ]:


doc = nlp(x)
for token in doc:
    print(token)


# In[ ]:





# ## Detecting Nouns 

# In[ ]:


x = 'Breaking News: Donal Trump, the president of the USA is looking to sign a deal to mine the moon'


# In[ ]:


doc = nlp(x)


# In[ ]:


for noun in doc.noun_chunks:
    print(noun)


# In[ ]:





# ## Language Translation and Detection

# Language Code: https://www.loc.gov/standards/iso639-2/php/code_list.php

# In[ ]:


x


# In[ ]:


tb = TextBlob(x)


# In[ ]:


tb.detect_language()


# In[ ]:


tb.translate(to = 'zh')


# In[ ]:





# ## Use TextBlob's Inbuilt Sentiment Classifier 

# In[ ]:


from textblob.sentiments import NaiveBayesAnalyzer


# In[ ]:


x = 'we all stands together. we are gonna win this fight'


# In[ ]:


tb = TextBlob(x, analyzer=NaiveBayesAnalyzer())


# In[ ]:


tb.sentiment


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




