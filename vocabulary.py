#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import nltk, re

def load_corpus(range):
    m = re.match(r'(\d+):(\d+)$', range)
    print 'm=', m
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
        from nltk.corpus import brown as corpus
	print corpus.fileids()
        return [corpus.words(fileid) for fileid in corpus.fileids()[start:end]]
#Need to change the load_corpus function
#Go through the entire corpus folder, call load_file for each file
#Read words from the file and return as wordlist
#Append wordlist to form a list of all words from a file

def load_file(filename):
    corpus = []
    f = open(filename, 'r')
    for line in f:
        doc = re.findall(r'\w+(?:\'\w+)?',line)
        if len(doc)>0:
            corpus.append(doc)
    f.close()
    return corpus

stopwords_list = nltk.corpus.stopwords.words('english')
#stopwords_list = "a,s,able,about,above,according,accordingly,across,actually,after,afterwards,again,against,ain,t,all,allow,allows,almost,alone,along,already,also,although,always,am,among,amongst,an,and,another,any,anybody,anyhow,anyone,anything,anyway,anyways,anywhere,apart,appear,appreciate,appropriate,are,aren,t,around,as,aside,ask,asking,associated,at,available,away,awfully,be,became,because,become,becomes,becoming,been,before,beforehand,behind,being,believe,below,beside,besides,best,better,between,beyond,both,brief,but,by,c,mon,c,s,came,can,can,t,cannot,cant,cause,causes,certain,certainly,changes,clearly,co,com,come,comes,concerning,consequently,consider,considering,contain,containing,contains,corresponding,could,couldn,t,course,currently,definitely,described,despite,did,didn,t,different,do,does,doesn,t,doing,don,t,done,down,downwards,during,each,edu,eg,eight,either,else,elsewhere,enough,entirely,especially,et,etc,even,ever,every,everybody,everyone,everything,everywhere,ex,exactly,example,except,far,few,fifth,first,five,followed,following,follows,for,former,formerly,forth,four,from,further,furthermore,get,gets,getting,given,gives,go,goes,going,gone,got,gotten,greetings,had,hadn,t,happens,hardly,has,hasn,t,have,haven,t,having,he,he,s,hello,help,hence,her,here,here,s,hereafter,hereby,herein,hereupon,hers,herself,hi,him,himself,his,hither,hopefully,how,howbeit,however,i,d,i,ll,i,m,i,ve,ie,if,ignored,immediate,in,inasmuch,inc,indeed,indicate,indicated,indicates,inner,insofar,instead,into,inward,is,isn,t,it,it,d,it,ll,it,s,its,itself,just,keep,keeps,kept,know,knows,known,last,lately,later,latter,latterly,least,less,lest,let,let,s,like,liked,likely,little,look,looking,looks,ltd,mainly,many,may,maybe,me,mean,meanwhile,merely,might,more,moreover,most,mostly,much,must,my,myself,name,namely,nd,near,nearly,necessary,need,needs,neither,never,nevertheless,new,next,nine,no,nobody,non,none,noone,nor,normally,not,nothing,novel,now,nowhere,obviously,of,off,often,oh,ok,okay,old,on,once,one,ones,only,onto,or,other,others,otherwise,ought,our,ours,ourselves,out,outside,over,overall,own,particular,particularly,per,perhaps,placed,please,plus,possible,presumably,probably,provides,que,quite,qv,rather,rd,re,really,reasonably,regarding,regardless,regards,relatively,respectively,right,said,same,saw,say,saying,says,second,secondly,see,seeing,seem,seemed,seeming,seems,seen,self,selves,sensible,sent,serious,seriously,seven,several,shall,she,should,shouldn,t,since,six,so,some,somebody,somehow,someone,something,sometime,sometimes,somewhat,somewhere,soon,sorry,specified,specify,specifying,still,sub,such,sup,sure,t,s,take,taken,tell,tends,th,than,thank,thanks,thanx,that,that,s,thats,the,their,theirs,them,themselves,then,thence,there,there,s,thereafter,thereby,therefore,therein,theres,thereupon,these,they,they,d,they,ll,they,re,they,ve,think,third,this,thorough,thoroughly,those,though,three,through,throughout,thru,thus,to,together,too,took,toward,towards,tried,tries,truly,try,trying,twice,two,un,under,unfortunately,unless,unlikely,until,unto,up,upon,us,use,used,useful,uses,using,usually,value,various,very,via,viz,vs,want,wants,was,wasn,t,way,we,we,d,we,ll,we,re,we,ve,welcome,well,went,were,weren,t,what,what,s,whatever,when,whence,whenever,where,where,s,whereafter,whereas,whereby,wherein,whereupon,wherever,whether,which,while,whither,who,who,s,whoever,whole,whom,whose,why,will,willing,wish,with,within,without,won,t,wonder,would,would,wouldn,t,yes,yet,you,you,d,you,ll,you,re,you,ve,your,yours,yourself,yourselves,zero".split(',')
recover_list = {"wa":"was", "ha":"has"}
wl = nltk.WordNetLemmatizer()

def is_stopword(w):
    return w in stopwords_list
def lemmatize(w0):
    w = wl.lemmatize(w0.lower())
    if w in recover_list: return recover_list[w]
    return w
def check_len(term):
    return len(term)<3

class Vocabulary:
    def __init__(self, excluds_stopwords=True):#False):
        self.vocas = []        # stores list of all unique words (vocabulary)
	#Dictionary vocas_id contains all the unique words (vocabulary) as the key
	#Id-Value is the sequence in which they were found.
        self.vocas_id = dict() # word to id, dictionary containing key_id:value = lemmatized_term which is not a Stopword and which is more than two characters long : count in the vocab list.
        self.docfreq = []      # id to document frequency
        self.excluds_stopwords = excluds_stopwords
        #print 'excluds_stopwords=', excluds_stopwords

    def term_to_id(self, term0):
	#Check if term present in the google dict_1gms, do this before lemmatizing
	#if not self.dict_1gms.has_key(term0):
	#  return None
        term = lemmatize(term0)
	#print 'TERM before lemmatize=', term0, 'after lemmatize=',term
        if not re.match(r'[a-z]+$', term):#Only terms which have alphabets after lemmatizing 
	    #print 'NOT match=', term
            return None 
	if check_len(term):
	    #print 'Very small word ', term
	    return None
        if self.excluds_stopwords and is_stopword(term): 
	    #print 'Stopword detect', term
	    return None
        if term not in self.vocas_id: #If term is a new term and is not already present in the vocabulary list
            voca_id = len(self.vocas)#Find the current length of the vocab list
            self.vocas_id[term] = voca_id #Add the term to the end of voca_id dictionary. value is the end of voca_list
            self.vocas.append(term) #Add term to vocabulary list
            self.docfreq.append(0)
	    #print 'TERM NOT in self.vocas_id', term, voca_id
        else:
            voca_id = self.vocas_id[term] #Term present in the vocabulary list. Return the vocabulary ID of the term.
	    #print 'PRESENT ',term 
        return voca_id

    def doc_to_ids(self, doc):
        #print ' '.join(doc)
        list = []
        words = dict()
        for term in doc:
            id = self.term_to_id(term) #Vocabulary id of the term, if term is not a stopword or small word < 3characters and contains only alphabets.
            if id != None:
                list.append(id) #List of vocabulary ids
                words[id] = 1 #Dictionary denoting words or terms which have been found in the document.
        if "close" in dir(doc): doc.close()
        for id in words: self.docfreq[id] += 1 #For all the words found in the document, increase their frequency
	#print 'doc=',doc,' list=', list
        return list

    def cut_low_freq(self, corpus, threshold=1):
        new_vocas = []
        new_docfreq = []
        self.vocas_id = dict()
        conv_map = dict()
        for id, term in enumerate(self.vocas):
            freq = self.docfreq[id]
            if freq > threshold:
                new_id = len(new_vocas)
                self.vocas_id[term] = new_id
                new_vocas.append(term)
                new_docfreq.append(freq)
                conv_map[id] = new_id
        self.vocas = new_vocas
        self.docfreq = new_docfreq

        def conv(doc):
            new_doc = []
            for id in doc:
                if id in conv_map: new_doc.append(conv_map[id])
            return new_doc
        return [conv(doc) for doc in corpus]

    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)

    def is_stopword_id(self, id):
        return self.vocas[id] in stopwords_list

