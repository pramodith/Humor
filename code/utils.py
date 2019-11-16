import pickle
import pandas as pd
import numpy as np
import spacy

def joke_file_prcessing():
   '''
   with open('../data/task-1/humorous_oneliners.pickle','rb') as f, open('../data/task-1/oneliners_incl_doubles.pickle','rb') as f1:
      obj =  pickle.load(f)
      obj.extend(pickle.load(f1))
   df = pd.DataFrame(obj,columns=["Joke"])
   df['Joke']=df['Joke'].apply(lambda x : x.replace("\n","").replace(",","").replace("\r",""))
   df.to_csv('shortjokes1.csv')
   '''

   with open('../data/task-1/reuters_headlines.pickle','rb') as f:
      obj = pickle.load(f)
      df = pd.read_csv('../data/task-1/shortjokes1.csv')
      df['label'] = 1
      df2 = pd.DataFrame(obj,columns=['Joke'])
      df2['label']  = 0
      df.drop(columns=['id'],inplace=True)
      df = pd.concat((df,df2),0,ignore_index=True)
      df.to_csv("joke_classification.csv")


def pos_tag(sentences, word):
   nlp = spacy.load("en_core_web_sm")
   pos = []
   sentences = [sent.replace("<","").replace("/>","") for sent in sentences]

   for ind,sentence in enumerate(sentences):
      doc =  nlp(sentence)
      for token in doc:
         if token.text==word[ind].strip("<|/>"):
            pos.append(token.pos_)
            break
      if len(pos)!=ind+1:
         pos.append("NOUN")
   return pos