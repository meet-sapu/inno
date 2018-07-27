import pandas as pd
#import numpy as np
from string import punctuation
#import tensorflow as tf
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
#import pickle 
import re
import disease_lookup


# Opening untagged sentences of both classes .

pos = open('/home/meet.sapariya9180/all_data/causality.pos')
pos = pos.read()    
pos = disease_lookup.clean_text(pos)
pos = pos.split('\n')

neg = open('/home/meet.sapariya9180/all_data/causality.neg')
neg = neg.read()
neg = disease_lookup.clean_text(neg)
neg = neg.split('\n')

# This function will tag the data and bring in a format appropriate for training .

def data_formation(pos_df,pos):    
    pos_df = pd.DataFrame(columns=['sentence','adv_eff','drug','disease'])
    drugs = []
    effs = []
    dss = []
    for txt in pos :
        ftxt = txt.lower()
        txt = ' '.join([k for k in ftxt.split() if k not in punctuation])
        res = disease_lookup.keywrd_finder(txt)
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

# Calling the above function on both classes .
    
neg_df = []
neg_df = data_formation(neg_df , neg)
pos_df = []
pos_df = data_formation(pos_df , pos)

neg_df['relation'] = 0
pos_df['relation'] = 1


# reading the ADE corpus .

with open('/home/meet.sapariya9180/all_data/ADE-Corpus-V2/ADE-NEG.txt') as ade_neg :
    adeneg = ade_neg.read()

ndata = adeneg.split('\n')
ndata = pd.DataFrame(ndata)
ndata[0] = ndata[0].apply(lambda x:re.sub(".+NEG","",x))

adn = []
adn = data_formation(adn,ndata[0])

final_data = disease_lookup.final_data

training_data = pd.concat([final_data,adn],ignore_index = True)
training_data = training_data.sample(frac=1)
training_data = training_data.reindex(range(0,len(training_data)))
training_data.to_csv('/home/meet.sapariya9180/relation_extraction_data.csv')

final_neg = pd.concat([neg_df,adn],ignore_index = True)    

final_data = final_data[['Sentence','Drug','adv_effect']]  
final_data = final_data.rename({'Sentence':'sentence','Drug':'drug','adv_effect':'adv_eff'},axis = 'columns')
final_data = final_data[['drug','adv_eff','sentence']]

final_pos = pd.concat([final_data,pos_df],ignore_index=True)
final_pos['relation'] = 1
final_neg['relation'] = 0


final_training_data = pd.concat([final_neg,final_pos],ignore_index = True)

# shuffling data .

final_training_data = final_training_data.sample(frac = 1)

final_training_data.to_csv('/home/meet.sapariya9180/big_relation_extraction_data.csv')




