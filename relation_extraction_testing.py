import pandas as pd
import numpy as np
from string import punctuation
import tensorflow as tf
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
import pickle 
import re
import disease_lookup
        
test_text = 'I ate crocin and I am having stomach infection '


def makePaddedList(sentences, pad_symbol= '<pad>'):
	maxl = max([len(sent) for sent in sentences])
	T = []
 	for sent in sentences:
		t = []
		lenth = len(sent)
		for i in range(lenth):
			t.append(sent[i])
		for i in range(lenth,maxl):
			t.append(pad_symbol)
		T.append(t)	

	return T, maxl


def testmakePaddedList(sentences, maxl,pad_symbol= '<pad>'):
	#maxl = max([len(sent) for sent in sentences])
	T = []
 	for sent in sentences:
		t = []
		lenth = len(sent)
		for i in range(lenth):
			t.append(sent[i])
		for i in range(lenth,maxl):
			t.append(pad_symbol)
		T.append(t)	
	return T, maxl



def preProcess(sent):
	sent = sent.lower()
	sent = tokenizer.tokenize(sent)
	sent = ' '.join([ s for s in sent if s not in punctuation])
	sent = re.sub('\d', 'dg',sent)
	#sent_list,_,_,_,_ = zip(*tagger.parse(sent)) 
	#sent = ' '.join(sent_list)
	return sent

#padded_sentences = makePaddedList(fdata['Sentence'])

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        print ind
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
  
import nltk    


def feature_making(sent_contents, drug, adv_effects):
    word_list = []
    d1_list = []
    d2_list= []
    type_list = []
    tagger_list = []           
    for sent, drug, adv_eff in zip(sent_contents, drug, adv_effects):
        sent = preProcess(sent)
        entity1 = preProcess(drug).split()
        entity2 = preProcess(adv_eff).split()
        s1,e1 = find_sub_list(entity1, list(sent.split()))
        s2,e2 = find_sub_list(entity2, list(sent.split()))
        tagger_list.append([tg[1] for tg in nltk.pos_tag(sent.split())])
        # distance1 feature	
        d1 = []
        for i in range(len(sent.split())):
    		    if i < s1 :
    			d1.append(str(i - s1))
    		    elif i > e1 :
    			d1.append(str(i - e1 ))
    		    else:
    			d1.append('0')
    		#distance2 feature		
        d2 = []
        for i in range(len(sent.split())):
    		    if i < s2:
    			d2.append(str(i - s2))
    		    elif i > e2:
    			d2.append(str(i - e2))
    		    else:
    			d2.append('0')
    		#type feature
        t = []
        for i in range(len(sent.split())):
    			t.append('Out')
        for i in range(s1, e1+1):
    			t[i] = 'drug'		
        for i in range(s2, e2+1):
    			t[i] = 'adv_eff'
                
        word_list.append(sent.split())
        d1_list.append(d1)
        d2_list.append(d2)
        type_list.append(t)     
        
    return word_list,d1_list, d2_list, type_list , tagger_list
    
def makeWordList(sent_list):
    wf = {}
    for sent in sent_list:
        for w in sent:
            if w in wf:
                wf[w] += 1
            else:
                wf[w] = 0
    
    wl = {}
    i = 0
    for w,f in wf.iteritems():		
        wl[w] = i
        i += 1
    wl['<UNK>'] = len(wl)
    return wl

def mapWordToId(sent_contents, word_dict):
    T = []
    k = 0
    for sent in sent_contents:
        t = []
        for w in sent:
            if w in word_dict.keys() :
                t.append(word_dict[w])
            else :
                t.append(word_dict['<UNK>'])
        T.append(t)
        print k
        k = k+1

    return T


def test_data_preprocess(test_text):
    pos_df = pd.DataFrame(columns=['sentence','adv_eff','drug','disease'])
    drugs = []
    effs = []
    dss = []
    txt = test_text
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
    
    pos_df['sentence'] = txt
    pos_df['adv_eff'] = effs
    pos_df['drug'] = drugs
    pos_df['disease'] = dss
    pos_df['lens'] = [len(dd) for dd in pos_df['drug'] ]
    pos_df = pos_df[pos_df['lens'] != 0]
    pos_df['lens'] = [len(dd) for dd in pos_df['adv_eff'] ]
    pos_df = pos_df[pos_df['lens'] != 0]
    del pos_df['disease']
    pos_df['sentence'] = txt
    del pos_df['lens']
    #pos_df = pos_df.drop_duplicates('sentence')
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

final_df = test_data_preprocess(test_text)

final_df = final_df.rename({'sentence':'Sentence','adv_eff':'adv_effect','drug':'Drug'},axis = 'columns')

word_list, d1_list, d2_list, type_list , tagger_list = feature_making(final_df['Sentence'],final_df['Drug'], final_df['adv_effect'])

#word_list, d1_list, d2_list, type_list , tagger_list = feature_making(fdata['Sentence'],fdata['Drug'], fdata['adv_effect'])

seq_len = 49

test_word_list, seq_len = testmakePaddedList(word_list,seq_len)
test_d1_list,_ = testmakePaddedList(d1_list,seq_len)
test_d2_list,_ = testmakePaddedList(d2_list,seq_len)
test_type_list,_ = testmakePaddedList(type_list,seq_len)
test_tagger_list,_ = testmakePaddedList(tagger_list,seq_len)

"""
test_word_dict = makeWordList(test_word_list)
test_d1_dict = makeWordList(test_d1_list)
test_d2_dict = makeWordList(test_d2_list)
test_type_dict = makeWordList(test_type_list)
test_tagger_dict = makeWordList(test_tagger_list)
"""

with open('/home/meet.sapariya9180/all_data/pickle_dicts_relation_extraction/big_word_dict.pickle','rb') as wp :
    test_word_dict = pickle.load(wp)
with open('/home/meet.sapariya9180/all_data/pickle_dicts_relation_extraction/big_d1_dict.pickle','rb') as dp :
    test_d1_dict = pickle.load(dp)
with open('/home/meet.sapariya9180/all_data/pickle_dicts_relation_extraction/big_d2_dict.pickle','rb') as dp :
    test_d2_dict = pickle.load(dp)
with open('/home/meet.sapariya9180/all_data/pickle_dicts_relation_extraction/big_type_dict.pickle','rb') as tp :
    test_type_dict = pickle.load(tp)
with open('/home/meet.sapariya9180/all_data/pickle_dicts_relation_extraction/big_tag_dict.pickle','rb') as tg :
    test_tagger_dict = pickle.load(tg)

test_W_train =  np.array(mapWordToId(test_word_list, test_word_dict))
test_d1_train = np.array(mapWordToId(test_d1_list, test_d1_dict))
test_d2_train = np.array(mapWordToId(test_d2_list, test_d2_dict))
test_T_train = np.array(mapWordToId(test_type_list,test_type_dict))
test_tag_train = np.array(mapWordToId(test_tagger_list,test_tagger_dict))



loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph('/home/meet.sapariya9180/all_data/saved_models_relation_extractoin/training_model.ckpt' + '.meta')
    loader.restore(sess, '/home/meet.sapariya9180/all_data/saved_models_relation_extractoin/training_model.ckpt')
    
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    w = loaded_graph.get_tensor_by_name('x:0')
    pos = loaded_graph.get_tensor_by_name('x1:0')
    d1 = loaded_graph.get_tensor_by_name('x2:0')
    d2 = loaded_graph.get_tensor_by_name('x3:0')
    typee = loaded_graph.get_tensor_by_name('x4:0')
    dropout_keep_prob = loaded_graph.get_tensor_by_name('dropout_keep_prob:0')
    ress = sess.run(logits , {w:test_W_train, pos : test_tag_train, d1: test_d1_train,d2:test_d2_train, typee:test_T_train,dropout_keep_prob:1.0})
    #ress = sess.run(logits,{w:np.reshape(test_W_train,(-1,97)), pos:np.reshape(test_tag_train,(-1,97)), d1: np.reshape(test_d1_train,(-1,97)),d2: np.reshape(test_d2_train,(-1,97)), typee:np.reshape(test_T_train,(-1,97)),dropout_keep_prob:1.0})




final_df['predictions'] = ress
from sklearn.metrics import accuracy_score , precision_score

accuracy_score(final_df['relation'],final_df['predictions'])

precision_score(final_df['relation'],final_df['predictions'])
