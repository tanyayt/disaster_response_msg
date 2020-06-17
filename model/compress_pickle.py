#import cPickle
import sys
import bz2
import pickle
import joblib

def load_compress(model_filename,compressed_filename):
    '''
    stores a compressed pickle file
    Args: the uncompressed model file name; and compressed pickle filename
    Returns: None 
    '''    
    model=joblib.load(model_filename)
    sfile = bz2.BZ2File(compressed_filename,"w")
    pickle.dump(model,sfile)

def tokenize(text):
    '''
    Args: text string
    Returns: a list of clean tokens
    '''
    #Replace urls with 'urlplaceholder' so urls are not tokenized
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    #find all urls and store in all_urls
    all_urls = re.findall(url_regex,text)
    
    for url in all_urls:
        text=text.replace(url,"urlplaceholder")
    
    #tokenize text
    tokens = word_tokenize(text)
    
    #remove stop words
    tokens =[token for token in tokens if token not in stopwords.words('english')]
    
    
    #crate a lemmatizer object 
    lemmatizer = WordNetLemmatizer()
    
    #clean tokens with lemmatizer
    clean_tokens =[]
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens

    
    
    
    
def main():
    model_filename,compressed_filename = sys.argv[1:]
    print("parameters taken-start compressing")
    load_compress(model_filename,compressed_filename)
    print("compressing finished")
        

if __name__ == "__main__":
    main()
    
        
        