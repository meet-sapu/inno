#!/usr/bin/env python2
"""
Created on Wed Jun  6 17:40:42 2018
@author: meet.sapariya9180
"""

import pandas as pd
#import sys
from flashtext import KeywordProcessor
from StringIO import StringIO
from pymongo import MongoClient 
#from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Reading various data .
disease_data = pd.read_json('~/all_data/disease.json')
with open('/home/meet.sapariya9180/all_data/ADE-Corpus-V2/DRUG-AE.rel','r') as d :
    data = d.read()

cdata = StringIO(data)
final_data = pd.read_csv(cdata,sep='|')
drugs_lookup = list(pd.unique(final_data['Drug']))
diseases = list(pd.unique(disease_data['name']))
data_adv = list(pd.unique(final_data['adv_effect']))

with open('/home/meet.sapariya9180/all_data/meddra_all_label_se.tsv') as sd :
    sider = sd.read()

tdata = StringIO(sider)
adv_data = pd.read_csv(tdata,sep='\t')
adv_effects = list(pd.unique(adv_data['Abdominal pain']))

kp_eff = KeywordProcessor()
fadv = adv_effects + data_adv
for add in list(pd.unique(fadv)):
    kp_eff.add_keyword(add.lower(),('Adv_effects',add.lower()))
    
kp_disease = KeywordProcessor()
for dis in diseases:
    kp_disease.add_keyword(dis.lower(),('Disease',dis.lower()))

def eff_finder(text):    
    return kp_eff.extract_keywords(text)

def disease_finder(text):
    return kp_disease.extract_keywords(text)

def drugs_name_from_db():
    conn = MongoClient('localhost',27017)
    database = 'db_name'
    db = conn[database]
    collection_name = 'collection_name'
    cursor = db[collection_name]
    coll = cursor.find({},{"synonyms":1,'_id':0})
    drug_names = []
    for data in coll:
        lenn = len(data[u'synonyms'])
        for i in range(0,lenn):
            drug_names.append(data[u'synonyms'][i][u'name_lower'])

def write_drug_name():
    with open('/home/meet.sapariya9180/all_data/drugs_names.txt', 'w') as file_handler:
        for drugs in drug_names:
            file_handler.write("{}\n".format(uni_code(drugs)))


drug_names = open('/home/meet.sapariya9180/all_data/drugs_names.txt')
drug_names = drug_names.read()
drug_names = drug_names.split('\n')
nd = ['human','amino acid','amino acids','allergy','can','body','result','it','reflex','level','umbilical cord','aim','mean','air','has','same','release','hand','h','the first','age','dose','vessels','age','degree','region','at 7','shape','control','light','outside']
drug_names = [n for n in drug_names if n not in nd]
kp_drug = KeywordProcessor()
for drug in drug_names:
    kp_drug.add_keyword(drug.lower(),('Drug',drug.lower()))

def drug_finder(text):
    return kp_drug.extract_keywords(text)
  

import unicodedata
def uni_code(text):
    return unicodedata.normalize('NFKD', text).encode('ascii','ignore')
            
        
def keywrd_finder(text):
     drugs = drug_finder(text)
     eff = eff_finder(text)
     disease = disease_finder(text)
     return drugs,eff,disease

from string import punctuation

def clean_text(text):
    return ''.join(s.lower() for s in text if s not in punctuation)

"""   
pos = open('/home/meet.sapariya9180/Downloads/causality.pos')
pos = pos.read()    
pos = clean_text(pos)
pos = pos.split('\n')

neg = open('/home/meet.sapariya9180/Downloads/causality.neg')
neg = neg.read()
neg = clean_text(neg)
neg = neg.split('\n')

def data_formation(pos_df,pos):    
    pos_df = pd.DataFrame(columns=['sentence','adv_eff','drug','disease'])
    drugs = []
    effs = []
    dss = []
    for txt in pos :
        ftxt = txt.lower()
        txt = ' '.join([k for k in ftxt.split() if k not in punctuation])
        res = keywrd_finder(txt)
        if len(res[0]) == 0 :
             drugs.append([])
        else :
            drugs.append(list(pd.DataFrame(res[0])[1]))
        if len(res[1]) == 0 :
             effs.append([])
        else :
            effs.append(list(pd.DataFrame(res[1])[1]))
        if len(res[2]) == 0 :
             dss.append([])
        else :
            dss.append(list(pd.DataFrame(res[2])[1]))
        
    pos_df['sentence'] = pd.DataFrame(pos)[0].apply(lambda x:' '.join([k for k in x.split()  if k not in punctuation ]))
    pos_df['adv_eff'] = effs
    pos_df['drug'] = drugs
    pos_df['disease'] = dss
    
    pos_df['lens'] = [len(dd) for dd in pos_df['drug'] ]
    pos_df = pos_df[pos_df['lens'] != 0]
    pos_df['lens'] = [len(dd) for dd in pos_df['adv_eff'] ]
    pos_df = pos_df[pos_df['lens'] != 0]
    
    del pos_df['disease']
    del pos_df['lens']
    
    pos_df = pos_df.drop_duplicates('sentence')
    pos_df['drug'] = pos_df['drug'].apply(lambda x:pd.unique(x))
    pos_df['adv_eff'] = pos_df['adv_eff'].apply(lambda x:pd.unique(x)) 
    pos_df['len_drug'] = [len(ld) for ld in pos_df['drug']]
    pos_df['len_ae'] = [len(ae) for ae in pos_df['adv_eff']]
    pos_df = pos_df.reset_index()
    del pos_df['index']
    rows = len(pos_df)
    final = pd.DataFrame()
    for i in range(0,rows):
        if int(pos_df['len_drug'][i]) == 1 and int(pos_df['len_ae'][i]) > 1:
            temp = pd.DataFrame()
            for j in range(0,int(pos_df['len_ae'][i])):
               temp = temp.append({ 'adv_eff':pos_df['adv_eff'][i][j],'sentence': pos_df['sentence'][i], 'drug': pos_df['drug'][i][0],'len_drug':1,'len_ae':1}, ignore_index=True)
            temp = temp[pos_df.columns]
            final = final.append(temp)
        if int(pos_df['len_drug'][i]) > 1 and int(pos_df['len_ae'][i]) == 1 :
            temp = pd.DataFrame()
            for j in range(0,int(pos_df['len_drug'][i])):
               temp = temp.append({ 'adv_eff':pos_df['adv_eff'][i][0],'sentence': pos_df['sentence'][i], 'drug': pos_df['drug'][i][j],'len_drug':1,'len_ae':1}, ignore_index=True)
            temp = temp[pos_df.columns]
            final = final.append(temp)
        if int(pos_df['len_drug'][i]) > 1 and int(pos_df['len_ae'][i]) > 1 :
             temp = pd.DataFrame()
             for k in range(0,int(pos_df['len_ae'][i])):
                 for j in range(0,int(pos_df['len_drug'][i])):
                     temp = temp.append({ 'adv_eff':pos_df['adv_eff'][i][k],'sentence': pos_df['sentence'][i], 'drug': pos_df['drug'][i][j],'len_drug':1,'len_ae':1}, ignore_index=True)
             temp = temp[pos_df.columns]
             final = final.append(temp)
        if int(pos_df['len_drug'][i]) == 1 and int(pos_df['len_ae'][i]) == 1 :
            final = final.append(pos_df.iloc[i])
        
        final['drug'] = final['drug'].apply(lambda x:x[0] if len(x)==1 else x)
        final['adv_eff'] = final['adv_eff'].apply(lambda x:x[0] if len(x)==1 else x)
        del final['len_drug']
        del final['len_ae']
    return final

text = "I am not feeling well from few days . I am having a cold from the past few days."

neg_df = []
neg_df = data_formation(neg_df , neg)
pos_df = []
pos_df = data_formation(pos_df , pos)

neg_df['relation'] = 0
pos_df['relation'] = 1

with open('/home/meet.sapariya9180/ADE-Corpus-V2/ADE-NEG.txt') as ade_neg :
    adeneg = ade_neg.read()

import re
ndata = adeneg.split('\n')
ndata = pd.DataFrame(ndata)
ndata[0] = ndata[0].apply(lambda x:re.sub(".+NEG","",x))

adn = []
adn = data_formation(adn,ndata[0])
"""
"""
adn = adn[['sentence','drug','adv_eff','relation']]
adn = adn.rename({'sentence':'Sentence','drug':'Drug','adv_eff':'adv_effect','relation':'relation'} , axis = 'columns')
"""
"""
training_data = pd.concat([final_data,adn],ignore_index = True)
training_data = training_data.sample(frac=1)
training_data = training_data.reindex(range(0,len(training_data)))
training_data.to_csv('/home/meet.sapariya9180/relation_extraction_data.csv')
"""
"""
final_neg = pd.concat([neg_df,adn],ignore_index = True)    

final_data = final_data[['Sentence','Drug','adv_effect']]  
final_data = final_data.rename({'Sentence':'sentence','Drug':'drug','adv_effect':'adv_eff'},axis = 'columns')
final_data = final_data[['drug','adv_eff','sentence']]

final_pos = pd.concat([final_data,pos_df],ignore_index=True)
final_pos['relation'] = 1
final_neg['relation'] = 0


final_training_data = pd.concat([final_neg,final_pos],ignore_index = True)

final_training_data = final_training_data.sample(frac = 1)

final_training_data.to_csv('/home/meet.sapariya9180/big_relation_extraction_data.csv')

"""

















