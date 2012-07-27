"""
Adapted from: Mathieu Blondel - 2010
Modifications made by: Tanushree Mitra
Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in
Finding scientifc topics (Griffiths and Steyvers)
"""

import numpy as np
import scipy as sp
from scipy import misc
from scipy.special import gammaln
import nltk, re
import vocabulary 
import os
import sys

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)

class LdaSampler(object):

    def __init__(self, n_topics, alpha, beta):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
	print 'alpha=', alpha	
	print 'beta=', beta 

    def _initialize(self, matrix):
        n_docs, vocab_size = matrix.shape

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics)) #document-topic count
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size)) #topic-term count
        self.nm = np.zeros(n_docs) #document-topic sum
        self.nz = np.zeros(self.n_topics) #topic-term sum
        self.topics = {}

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics) #random initialization of the 1st topic
                self.nmz[m,z] += 1 #increment document-topic count
                self.nm[m] += 1 #increment document-topic sum
                self.nzw[z,w] += 1 #increment topic-term count
                self.nz[z] += 1 #increment topic-term sum
                self.topics[(m,i)] = z #topics assigned to words of document.(Row=document index, Col=word/vocab index)

    def _conditional_distribution(self, m, w):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.nzw.shape[1]
	left = (self.nzw[:,w] + self.beta) / \
               (self.nz + self.beta * vocab_size) #Vector element-wise division
        right = (self.nmz[m,:] + self.alpha) / \
                (self.nm[m] + self.alpha * self.n_topics)
        p_z = left * right
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzw.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in xrange(self.n_topics):
            lik += log_multi_beta(self.nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = self.nzw.shape[1]
        num = self.nzw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def output_word_topic_dist(self, filename, vocab, words_pertopic):
	"""
	Output word topic distribution
	"""
	#Num of times topic z and word w co-occur nzw[:,w]. Change it into probabilities by dividing with topic term sum nz
	Ftopic = open(outfolder+'/topic_word_'+filename, 'w')
	for z in xrange(self.n_topics):
		word_dist = (self.nzw[z,:]) / \
			(self.nz[z])
		print '\ntopic = ', z, '\n-----'
		Ftopic.write('\ntopic = '+ str(z) + '\n-----')
		for i in np.argsort(-word_dist)[:words_pertopic]: #descending order sort and return index 'i' for the top 20 terms with highest probability
		  print vocab[i], word_dist[i] #print word and its probability for the topic
		  Ftopic.write(vocab[i] + ' ' + str(word_dist[i]) + '\n')
	Ftopic.close()

    def output_doc_topic_dist(self, filename, vocab, doc_list):
	"""
	Output document topic distribution
	"""
	#Num of times document m and topic z co-occur nmz[:,z]. Change it into probabilities by dividing with document-topic sum nm
	Fdoc_topic = open(outfolder+'/doc_topic_'+filename, 'w')
	print 'document list length=', len(doc_list)
	for m in xrange(len(doc_list)):
		topic_dist = (self.nmz[m,:]) / \
			(self.nm[m])
		print 'document = ', doc_list[m],
		Fdoc_topic.write('\ndocument = '+ doc_list[m])
		Fdoc_topic.write('[')
		for i in np.argsort(-topic_dist):
		  print 'topic',i, topic_dist[i],',', 
		  Fdoc_topic.write('topic'+str(i) + ' ' + str(topic_dist[i])+',')
		Fdoc_topic.write(']\n')
		print '\n'
	Fdoc_topic.close()

    def run(self, matrix, maxiter):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = matrix.shape
        self._initialize(matrix)

        for it in xrange(maxiter):
            for m in xrange(n_docs):
                for i, w in enumerate(word_indices(matrix[m, :])):
                    z = self.topics[(m,i)]
                    self.nmz[m,z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z,w] -= 1
                    self.nz[z] -= 1

                    p_z = self._conditional_distribution(m, w)
                    z = sample_index(p_z)

                    self.nmz[m,z] += 1
                    self.nm[m] += 1
                    self.nzw[z,w] += 1
                    self.nz[z] += 1
                    self.topics[(m,i)] = z

            # FIXME: burn-in and lag!
            yield self.phi()

if __name__ == "__main__":
    import os
    import shutil
    import optparse

    parser = optparse.OptionParser("Example usage:-\n python lda_gibbs.py -f input_tweet1000.data -k 18 -a 0.05 -b 0.05 -i 10 -w 10 -o output_folder\n where, f=inputfile \n k=num_topics \n a=alpha \n b=beta \n i=maximum iterations \n w=number of words per topic to output in topic_word_dist \n o=output folder\n")
    parser.add_option("-f","--file", dest="filename", help="filename is the input document")
    parser.add_option("-k","--n_topics", dest="n_topics", help="Number of topics")
    parser.add_option("-a","--alpha", dest="alpha", help="alpha")
    parser.add_option("-b","--beta", dest="beta", help="beta")
    parser.add_option("-i","--num_iter", dest="num_iter", help="num_iter")
    parser.add_option("-w","--words_pertopic", dest="words_pertopic", help="words_pertopic")
    parser.add_option("-o","--out_folder", dest="out_folder", help="out_folder")
    (options, args) = parser.parse_args()
    print 'len(args)=', len(args), 'args=',args, 'options=',options

    N_TOPICS = int(options.n_topics) #10
    alpha = float(options.alpha)
    beta = float(options.beta)
    num_iter = int(options.num_iter)
    outfolder = options.out_folder
    words_pertopic = int(options.words_pertopic) 

    if(options.filename == None or options.n_topics == None):
      print 'Usage: python lda_gibbs.py -f filename -k num_topics'
      print '-----------------------------------------\n'
      parser.errors("Incorrect Usage")  

    def load_file(filename):
      """ Each line of file corresponds to a document
	  Extract each word of each line (aka document).
	  Make a list of words in the document
	  Append the list to corpus. Corpus is a list of documents """
      corpus = [] #corpus = [[word_doc1, word_doc1,..], [word_doc2, word_doc2, ....]....]
      print 'FILENAME=', filename
      f = open(filename, 'r')
      for line in f:
        doc = re.findall(r'\w+(?:\'\w+)?',line)
        if len(doc)>0:
            corpus.append(doc)
      f.close()
      return corpus
    
    def gen_vocab_terms(docs):
      """Finding the list of unique words (vocabulary) """
      vocab = []
      for doc in docs:
  	for word in doc:
  	  if word not in vocab:
  	    vocab.append(word)
      return vocab

    def gen_doc_matrix(docs, vocab):
      """ Creating document term matrix
      matrix = -------------- VOCABULARY TERMS ------------
      	      |       term1 term2 term3.....
	      |  doc1  5     1      0  ..... occurrences
	      D  doc2  0     2      4  ......occurrences
	      O
	      C
	      S
	      |
      """
      matrix = np.zeros((len(docs),len(vocab))) 
      doc_num = -1
      for doc in docs:
        doc_num += 1
        for term_id in vocab:
  	  if term_id in doc:
  	      matrix[doc_num][term_id] += 1
	      #print doc_num,' ', term_id,
      return matrix
     
    print 'Generating word dist \n'
    sampler = LdaSampler(N_TOPICS,alpha,beta)
   
    """Read the documents from file and save them in a list
    This will be used in the output_doc_topic_dist module"""
    doc_list = []
    f = open(options.filename, 'r')
    while(1):
      line = f.readline()
      if not line:
        break
      doc_list.append(line)
  
    print 'Loading file -- '
    corpus = load_file(options.filename)#'tweet_combin_n10.data')#('tweet_eng_25jan11_trim.data')#('input.data')#('tweet_eng_25jan2011.data')
    voca = vocabulary.Vocabulary(True)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    vocab = voca.vocas
    print 'vocabulary length = ', len(vocab)
    #FILE_vocab = open('output_raterCalais/vocab'+options.filename,'w')
    FILE_vocab = open(outfolder+'/vocab'+options.filename,'w')
    FILE_vocab.write('Vocabulary length = ' + str(len(vocab))+'\n')
    for i in vocab:
      FILE_vocab.write(i+'\n')
    FILE_vocab.close()
    vocab_id = gen_vocab_terms(docs) #I could have used the code in Vocabulary.py but this works. 
   
    matrix = gen_doc_matrix(docs, vocab_id)

    Fout = open(outfolder+'/loglike-'+options.filename,'w')
    Fout.write('Iteration log-liklihood\n')
    for it, phi in enumerate(sampler.run(matrix,num_iter)):
        print "Iteration", it + 1
	ll = sampler.loglikelihood()
        print "Likelihood", ll
	Fout.write(str(it+1) + ' ' + str(ll) +'\n')
    sampler.output_word_topic_dist(options.filename, vocab,words_pertopic)
    sampler.output_doc_topic_dist(options.filename, vocab, doc_list)
    Fout.close()
