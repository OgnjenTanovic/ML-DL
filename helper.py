#----------------Web scraping-----------
import urllib.request as url
from bs4 import BeautifulSoup
import re

import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer



def get_urls(main_url):
    resp = url.urlopen(main_url)#open url
    html_doc = resp.read()#read page
    soup = BeautifulSoup(html_doc, 'html.parser')
    elements = soup.find_all('section')#find tag section
    # elements = soup.find_all('div', {'class':'pagination'}) search by tag and class
    # element.find_all(['h3', 'figure']) search multiple tags
    '''
    Remove unwonted tags
    >>> s=[]
    >>> for i, element in enumerate(elements):
    ...     s.append(element)
    ...     unw = s[i].find_all(['figure', 'script', 'span', 'ins'])
    ...     for u in unw:
    ...         u.extract()
    '''
    urls = []
    with open('urls.txt', 'a') as u:
        for element in elements:        
            temp = element.find_all('a')
            for x in temp:
                if not x.get('href') in urls:
                    urls.append(x.get('href'))
                    u.write(x.get('href') + "\n")
                    get_page_text(x.get('href'))
                    
def get_page_text(link):    
    resp = url.urlopen(link)#open url
    html_doc = resp.read()#read page
    soup = BeautifulSoup(html_doc, 'html.parser')
    txt = soup.find_all('h3')# find elements with text
    #read text from elements
    temp = []
    for e in txt:
        temp.append(e.text)
    # join text from elements
    text = ' '.join(text for text in temp)
    
STOPWORDS = set(stopwords.words('english'))
porter = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\']", " ", text)#remove averything except characters and \'
    text = re.sub(r"\'s", "", text)#remove It's, That's...
    text = re.sub(r"\'re", "", text)#remove They're, you're...
    text = re.sub(r"n\'", " not", text)#remove does'nt, isn't...
    text = ' '.join(txt for txt in text.split() if txt not in STOPWORDS)#delete stopwors from text
    #lemmatization, converts the word into its root word
    text = ' '.join(wordnet_lemmatizer.lemmatize(word) for word in text.split())
    #stemming, removal of suffices, like “ing”, “ly”, “s”
    text = ' '.join(porter.stem(word) for word in text.split())
    #
    return text
    
    
    
    
    
    
    
    
    
    
    
    

    
