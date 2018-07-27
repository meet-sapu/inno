import pandas as pd
import numpy as np
from string import punctuation
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()


# This data is prepared by using script disease_lookup .
tdata = pd.read_csv('/home/meet.sapariya9180/big_relation_extraction_data.csv')
tdata = tdata.rename({'sentence':'Sentence','drug':'Drug','adv_eff':'adv_effect'},axis = 'columns')
tdata = tdata[['Unnamed: 0', 'Sentence', 'Drug', 'adv_effect', 'relation']]
del tdata['Unnamed: 0']


# This function is written for data augmentation , i.e it will change the adverse effects in the sentences of a particular drug .
"""
The data augmentration result was not good .
"""
def data_augmentation(tdata):
    gg = tdata.groupby('Drug')
    gg = gg.groups
    training_final = pd.DataFrame()
    for key in gg.keys():
        temp = tdata[tdata['Drug']==key]
        cc = {}
        for aa in pd.unique(temp['adv_effect']):
            ttemp = temp[temp['adv_effect']==aa]
            if len(pd.unique(ttemp['relation'])) > 1:
                cc[aa] = 1
            else :
                cc[aa] = 0
        temp['flag'] = temp['adv_effect'].map(cc)
        temp = temp[temp['flag'] != 1]
        del temp['flag']
        #ftemp = pd.DataFrame(columns=temp.columns)
        sentence = []
        relation = []
        adv_eff = []
        flist = []
        if len(pd.unique(temp['relation'])) == 2:
            for i in range(0,len(temp)):
                for j in range(0,len(temp)):
                    if i != j :
                        sentence.append(temp['Sentence'].iloc[i].replace(temp['adv_effect'].iloc[i],temp['adv_effect'].iloc[j]))
                        adv_eff.append(temp['adv_effect'].iloc[j])
                        relation.append(temp['relation'].iloc[j])
            for i in range(0,len(sentence)):
                flist.append([sentence[i],adv_eff[i],relation[i]])
            for i in range(0,len(temp)):
                flist.append([temp['Sentence'].iloc[i],temp['adv_effect'].iloc[i],temp['relation'].iloc[i]])
            atemp = temp[temp['relation']==1]
            atemp = atemp[:len(atemp)/2]
            #adv_hc = pd.unique(atemp['adv_effect'])
            
            etemp = temp[temp['relation']==0]
            edv_hc = pd.unique(etemp['adv_effect'])
            el = []
            for ed in edv_hc:
                el.append([key+' has cured '+ed,ed,0])
                
            ltemp = []
            ltemp = flist
            final = []
            for i in range(0,len(ltemp)):
                if flist[0] not in final :
                    final.append(flist[0])
                del flist[0]
            final = pd.DataFrame(final)
            final = final.rename(columns={0: "Sentence", 1: "adv_effect",2:"relation"})
            final['Drug'] = key
            final = final[temp.columns]
            training_final = training_final.append(final)
            training_final = training_final.sample(frac=1).reset_index(drop=True)
        if len(pd.unique(temp['relation'])) == 1 and pd.unique(temp['relation']) == 0 :
            etemp = temp[temp['relation']==0]
            edv_hc = pd.unique(etemp['adv_effect'])
            el = []
            for ed in edv_hc:
                el.append([key+' has cured '+ed,ed,0])
            final = pd.DataFrame(el)
            final = final.rename(columns={0: "Sentence", 1: "adv_effect",2:"relation"})
            final['Drug'] = key
            final = final[temp.columns]
            temp = temp.append(final)
            training_final = training_final.append(temp)    
            training_final = training_final.sample(frac=1).reset_index(drop=True)
        print key
            
    tdata = training_final    
    tdata.to_csv('/home/meet.sapariya9180/augmented_relation_data.csv')
    

# This will pad the sentences to a fxed number , used for training data .
    
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

tdata['length'] = [ len(sent.split()) for sent in tdata['Sentence'] ]

import re
#from string import punctuation

#Removing all the sentences from the training data less than 40 .

fdata = tdata[tdata['length'] < 40 ]
fdata = fdata.reset_index()
del fdata['index']
ff = []
for i in range(0,len(fdata)):
    if fdata['adv_effect'][i].lower() in fdata['Sentence'][i].lower().split() and fdata['Drug'][i].lower() in fdata['Sentence'][i].lower().split() :
        ff.append(1)
    else :
        ff.append(0)

fdata['ff'] = ff

fdata = fdata[fdata['ff'] == 1]
del fdata['ff']

# removing punctuations from every sentence .

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


# Will make feature of every sentence in the training data required for training .    

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


# The 5 features used for training . 

word_list, d1_list, d2_list, type_list , tagger_list = feature_making(fdata['Sentence'],fdata['Drug'], fdata['adv_effect'])

#padding

word_list, seq_len = makePaddedList(word_list)
d1_list,_ = makePaddedList(d1_list)
d2_list,_ = makePaddedList(d2_list)
type_list,_ = makePaddedList(type_list)
tagger_list,_ = makePaddedList(tagger_list)


# giving a unique number to every number and word in the feature .

word_dict = makeWordList(word_list)
d1_dict = makeWordList(d1_list)
d2_dict = makeWordList(d2_list)
type_dict = makeWordList(type_list)
tagger_dict = makeWordList(tagger_list)


def pickle_dicts(word_dict,d1_dict,d2_dict,type_dict,tagger_dict):
    with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_aug_word_dict.pickle','wb') as wp :
        pickle.dump(word_dict,wp)
    with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_aug_d1_dict.pickle','wb') as dp :
        pickle.dump(d1_dict,dp)
    with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_aug_d2_dict.pickle','wb') as dp :
        pickle.dump(d2_dict,dp)
    with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_aug_type_dict.pickle','wb') as tp :
        pickle.dump(type_dict,tp)
    with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_aug_tag_dict.pickle','wb') as tg :
        pickle.dump(tagger_dict,tg)


# Saving all the dictionaries , as these will be used again at the time of training .

pickle_dicts(word_dict,d1_dict,d2_dict,type_dict,tagger_dict)

"""
def read_pickle_dicts():
    with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_word_dict.pickle','r') as wp :
        word_dict = pickle.load(wp)
    with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_d1_dict.pickle','r') as dp :
        d1_dict = pickle.load(dp)
    with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_d2_dict.pickle','r') as dp :
        d2_dict = pickle.load(dp)
    with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_type_dict.pickle','r') as tp :
        type_dict = pickle.load(tp)
    with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_tag_dict.pickle','r') as tg :
        tagger_dict = pickle.load(tg)
"""

import pickle


# Mapping of the features and saving them .

W_train =  np.array(mapWordToId(word_list, word_dict))
with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/W_train_big.pickle','wb') as wp :
    pickle.dump(W_train,wp)

d1_train = np.array(mapWordToId(d1_list, d1_dict))
with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/d1_train_big.pickle','wb') as wp :
    pickle.dump(d1_train,wp)

d2_train = np.array(mapWordToId(d2_list, d2_dict))
with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/d2_train_big.pickle','wb') as wp :
    pickle.dump(d2_train,wp)

T_train = np.array(mapWordToId(type_list,type_dict))
with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/T_train_big.pickle','wb') as wp :
    pickle.dump(T_train,wp)

P_train = np.array(mapWordToId(tagger_list,tagger_dict))
with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/P_train_big.pickle','wb') as wp :
    pickle.dump(P_train,wp)

# converting into array .

W_train = np.array(W_train)
d1_train = np.array(d1_train)
d2_train = np.array(d2_train)
T_train = np.array(T_train)
P_train = np.array(P_train)

fdata = fdata.reset_index()
del fdata['index']

# Splitting into training and test data .

from sklearn.cross_validation import train_test_split
y_train , y_test = train_test_split(fdata['relation'] , test_size =0.25 , random_state = 42 ,stratify = fdata['relation'])
Y_train = pd.DataFrame(columns = [0,1])
Y_train[0] = fdata['relation']
Y_train[1] = fdata['relation']

map_func = {1:0 , 0:1}
Y_train[0] = Y_train[0].map(map_func)

word_dict_size = len(word_dict)
d1_dict_size = len(d1_dict)
d2_dict_size = len(d2_dict)
type_dict_size = len(type_dict)
tagger_dict_size = len(tagger_dict)
label_dict_size = len(map_func)


# importing word2vec and and saving into .txt format .

def saving_emb_bin_txt():
    from gensim.models import word2vec
    model = word2vec.Word2Vec.load_word2vec_format('/home/meet.sapariya9180/Downloads/PubMed-w2v.bin', binary=True)
    model.save_word2vec_format('/home/meet.sapariya9180/PubMed-w2v.txt', binary=False)



# reading words which are only used in the vocab of the data .

def readWordEmb(word_dict, fname, embSize=200):
    print "Reading word vectors"
    fwords = word_dict.keys()
    wordemb = {}
    with open(fname, 'r') as f:
        count = 0 
        for line in f :			
            vs = line.split()
            if vs[0] in fwords :
                wordemb[vs[0]] = map(float, vs[1:])
        
        #wordemb11 = np.asarray(wordemb, dtype='float32')
	print "number of unknown word in word embedding", count
	return wordemb
			
import pickle
def saving_pickle_wrd_vec(ww):
    with open('/home/meet.sapariya9180/pubmed.pickle','wb') as f:
        pickle.dump(ww,f)


# Mapping of word embedding used in the training data . Run this only once , takes very long .

wv = readWordEmb(word_dict, '/home/meet.sapariya9180/PubMed-w2v.txt')
saving_pickle_wrd_vec(wv)

# Reading saved mapping .

with (open("/home/meet.sapariya9180/pubmed.pickle", "rb")) as openfile:
       data = pickle.load(openfile)        

wv = data

# initializing the words vectors randomly whose mapping is not present in the word2vec dictionary .

for wrd in word_dict.keys():
    if wrd not in wv.keys():
        wv[wrd] = np.random.rand(200)

final_wrd_emb = []
for key in wv.keys():
    final_wrd_emb.append(wv[key])

final_wrd_emb = np.array(final_wrd_emb)

#W_train =  np.array(mapWordToId(word_list, word_dict))
#P_train = np.array(mapWordToId(tagger_list, tagger_dict))
#d1_train = np.array(mapWordToId(d1_list, d1_dict))
#d2_train = np.array(mapWordToId(d2_list, d2_dict))
#T_train = np.array(mapWordToId(type_list,type_dict))

new = {}

for k in word_dict.keys():
    new[k] = wv[k]

tr = np.array(y_train.index)
te = np.array(y_test.index)

Y_train = np.array(Y_train)

# Splitting of the features for training and testing .

W_tr, W_te = W_train[tr], W_train[te]
P_tr, P_te = P_train[tr], P_train[te]
d1_tr, d1_te = d1_train[tr], d1_train[te]
d2_tr, d2_te = d2_train[tr], d2_train[te]
T_tr, T_te = T_train[tr], T_train[te]
Y_tr, Y_te = Y_train[tr], Y_train[te]
Y_tr = np.array(Y_tr)
Y_te = np.array(Y_te)

nwv = []
for key in new.keys():
    nwv.append(new[key])

nwv = np.array(nwv)
nwv = nwv.astype('float32')

with open('/home/meet.sapariya9180/saved_models_relation_extractoin/final_embedding.txt','wb') as g:
    pickle.dump(nwv,g)
    
import tensorflow as tf

# number of classes .

num_classes = 2

class CNN_Relation(object):

    def __init__(self, num_classes, seq_len, word_dict_size, tagger_dict_size , d1_dict_size, d2_dict_size, type_dict_size, wv, batch_size = 200, w_emb_size=200, d1_emb_size=5, d2_emb_size=5, pos_emb_size=5, dep_emb_size=5, type_emb_size=5, filter_sizes=[3,5,7], num_filters=100, l2_reg_lambda = 0.01):
        emb_size = w_emb_size + pos_emb_size + d1_emb_size + d2_emb_size + type_emb_size  
#		emb_size = w_emb_size + d1_emb_size + d2_emb_size + type_emb_size  
#		emb_size = w_emb_size + type_emb_size  
        self.w  = tf.placeholder(tf.int32, [None, seq_len], name="x")
        self.pos = tf.placeholder(tf.int32, [None, seq_len], name="x1")
        self.d1 = tf.placeholder(tf.int32, [None, seq_len], name="x2")
        self.d2 = tf.placeholder(tf.int32, [None, seq_len], name='x3')
        self.typee = tf.placeholder(tf.int32, [None, seq_len], name='x4')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		# Initialization
		#W_wemb =    tf.Variable(tf.random_uniform([word_dict_size, w_emb_size], -1.0, +1.0))
        W_wemb = tf.Variable(nwv)
        W_posemb =  tf.Variable(tf.random_uniform([tagger_dict_size, pos_emb_size], -1.0, +1.0))
        W_d1emb =   tf.Variable(tf.random_uniform([d1_dict_size, d1_emb_size], -1.0, +1.0))
        W_d2emb =   tf.Variable(tf.random_uniform([d2_dict_size, d2_emb_size], -1.0, +1.0))
        W_typeemb = tf.Variable(tf.random_uniform([type_dict_size, type_emb_size], -1.0, +1.0))		
		# Embedding layer
        emb0 = tf.nn.embedding_lookup(W_wemb, self.w)				#word embedding
        emb1 = tf.nn.embedding_lookup(W_posemb, self.pos)			#position from first entit
        emb2 = tf.nn.embedding_lookup(W_d1emb, self.d1)				#POS embedding
        emb3 = tf.nn.embedding_lookup(W_d2emb, self.d2)				#POS embedding
        emb4 = tf.nn.embedding_lookup(W_typeemb, self.typee)			#POS embedding 
        X = tf.concat([emb0, emb1, emb2, emb3, emb4],2)			#shape(?, 21, 80)
#		X = tf.concat(2, [emb0, emb3, emb4, emb5])
#		X = tf.concat(2, [emb0, emb5])
        X_expanded = tf.expand_dims(X, -1) 					#shape (?, 21, 80, 1)

        l2_loss = tf.constant(0.0)
		
		# CNN+Maxpooling Layer
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, emb_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(X_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
        		# Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 		#shape (?, 19, 1, 70)
			# print "h ", h.get_shape
			# Maxpooling over the outputs
            pooled = tf.nn.max_pool(h, ksize=[1, seq_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
			# print "pooled", pooled.get_shape				#shape=(?, 1, 1, 70)
            pooled_outputs.append(pooled)

		#print "pooled_outputs", len(pooled_outputs)

		# Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs , 3)					#shape= (?, 1, 1, 210)
		#print "h_pool", h_pool.get_shape()
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])		#shape =(?, 210)
		#print "h_pool_flate", h_pool_flat.get_shape
		# dropout layer	 
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
		# Fully connetected layer
        W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
        #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        #l2_loss += tf.nn.l2_loss(b)
        scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
		# prediction and loss function
        self.predictions = tf.argmax(scores, 1, name="predictions")
        self.losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = scores))
        self.losses = tf.reduce_mean(self.losses + l2_reg_lambda * l2_loss)
        	# Accuracy
        self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")	 
		#session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session()  
        self.optimizer = tf.train.AdamOptimizer(1e-2)        
        self.grads_and_vars = self.optimizer.compute_gradients(self.losses)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
        self.sess.run(tf.initialize_all_variables())

    def train_step(self, W_batch, pos_batch, d1_batch, d2_batch, t_batch, y_batch):
        feed_dict = {self.w:W_batch,self.pos	:pos_batch, self.d1:d1_batch,self.d2:d2_batch,self.typee:t_batch,self.dropout_keep_prob: 1.0,self.input_y:y_batch}
        _, step, loss, accuracy, predictions = self.sess.run([self.train_op, self.global_step, self.losses, self.accuracy, self.predictions], feed_dict)
        print ("step "+str(step) + " loss "+str(loss) +" accuracy "+str(accuracy))
    
    def saving_tf(self,checkpoint) :
        saver = tf.train.Saver()
        saver.save(self.sess , checkpoint)
      
    def test_step(self, W_batch, pos_batch,  d1_batch, d2_batch, t_batch, y_batch):
        feed_dict = {self.w:W_batch,self.pos:pos_batch, self.d1:d1_batch,self.d2 	:d2_batch,self.typee	:t_batch,self.dropout_keep_prob:1.0,self.input_y:y_batch}
        step, loss, accuracy, predictions = self.sess.run([self.global_step, self.losses, self.accuracy, self.predictions], feed_dict)
        print "Accuracy in test data", accuracy
        return accuracy, predictions


print 'Training about to start .'
seq_len = 49
cnn = CNN_Relation(label_dict_size, seq_len, word_dict_size, tagger_dict_size , d1_dict_size, d2_dict_size, type_dict_size, nwv)


# Hyperparameter initialization .

num_train = len(W_tr)
y_true_list = []
y_pred_list = []
num_epochs = 30
N = 20
batch_size=128
num_batches_per_epoch = int(num_train/batch_size) + 1

def test_step(W_te, P_te, d1_te, d2_te, T_te, Y_te):
	n = len(W_te)	 
	num = int(n/batch_size) + 1
	sample = []
	for batch_num in range(num):	
		start_index = batch_num*batch_size
		end_index = min((batch_num + 1) * batch_size, n)
		sample.append(range(start_index, end_index))
	#acc = [] 
	pred = []
	for i in sample:
		a,p = cnn.test_step(W_te[i], P_te[i],d1_te[i], d2_te[i],T_te[i], Y_te[i])
#		acc.extend(a)
		pred.extend(p)
	return pred


for j in range(num_epochs):		
    print "epoch=%s"%j
    sam=[]
    for batch_num in range(num_batches_per_epoch):	
        start_index = batch_num*batch_size
        end_index = min((batch_num + 1) * batch_size, num_train)
        sam.append(range(start_index, end_index))
    for rang in sam:
        cnn.train_step(W_tr[rang], P_tr[rang], d1_tr[rang], d2_tr[rang], T_tr[rang], Y_tr[rang])
        
    if (j%N) == 0:
        cnn.saving_tf('/home/meet.sapariya9180/finance_classifier/bb_training_model.ckpt')
        pred = test_step(W_te, P_te, d1_te, d2_te, T_te, Y_te)			 
        print "test data size ", len(pred)
        y_true = np.argmax(Y_te, 1)
        y_pred = pred
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
    
"""    
#Testing#
        
test_text = 'I ate crocin and I am having stomach infection '

def test_data_preprocess(test_text):
    pos_df = pd.DataFrame(columns=['sentence','adv_eff','drug','disease'])
    drugs = []
    effs = []
    dss = []
    txt = test_text
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
fdata = fdata.reset_index()
del fdata['index']
final_df = fdata.iloc[te]

final_df = final_df.rename({'sentence':'Sentence','adv_eff':'adv_effect','drug':'Drug'},axis = 'columns')
final_df = final_df.iloc[0:5000]

word_list, d1_list, d2_list, type_list , tagger_list = feature_making(final_df['Sentence'],final_df['Drug'], final_df['adv_effect'])

#word_list, d1_list, d2_list, type_list , tagger_list = feature_making(fdata['Sentence'],fdata['Drug'], fdata['adv_effect'])

test_word_list, seq_len = testmakePaddedList(word_list,seq_len)
test_d1_list,_ = testmakePaddedList(d1_list,seq_len)
test_d2_list,_ = testmakePaddedList(d2_list,seq_len)
test_type_list,_ = testmakePaddedList(type_list,seq_len)
test_tagger_list,_ = testmakePaddedList(tagger_list,seq_len)
"""
"""
test_word_dict = makeWordList(test_word_list)
test_d1_dict = makeWordList(test_d1_list)
test_d2_dict = makeWordList(test_d2_list)
test_type_dict = makeWordList(test_type_list)
test_tagger_dict = makeWordList(test_tagger_list)
"""
"""

with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_word_dict.pickle','rb') as wp :
    test_word_dict = pickle.load(wp)
with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_d1_dict.pickle','rb') as dp :
    test_d1_dict = pickle.load(dp)
with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_d2_dict.pickle','rb') as dp :
    test_d2_dict = pickle.load(dp)
with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_type_dict.pickle','rb') as tp :
    test_type_dict = pickle.load(tp)
with open('/home/meet.sapariya9180/pickle_dicts_relation_extraction/big_tag_dict.pickle','rb') as tg :
    test_tagger_dict = pickle.load(tg)

test_W_train =  np.array(mapWordToId(test_word_list, test_word_dict))
test_d1_train = np.array(mapWordToId(test_d1_list, test_d1_dict))
test_d2_train = np.array(mapWordToId(test_d2_list, test_d2_dict))
test_T_train = np.array(mapWordToId(test_type_list,test_type_dict))
test_tag_train = np.array(mapWordToId(test_tagger_list,test_tagger_dict))




import tensorflow as tf


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph('/home/meet.sapariya9180/saved_models_relation_extractoin/training_model.ckpt' + '.meta')
    loader.restore(sess, '/home/meet.sapariya9180/saved_models_relation_extractoin/training_model.ckpt')
    
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

"""






