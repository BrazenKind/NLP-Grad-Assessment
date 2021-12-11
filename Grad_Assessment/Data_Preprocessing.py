#!/usr/bin/env python
# coding: utf-8

#Original ipynb file under Redundant Code folder

# In[3]:


import numpy as np
import regex as re


# In[19]:


class Data_Preprocessing:
    def __init__(self, x_data, y_data, x_lemmas, x_vocab, labels, label_dict):
        
        #x_data: the sentence we're converting into features
        #y_data: the labels that correspond to the sentence we're converting
        #x_lemmas, the lemma forms of each word within the sentence we're converting
        #x_vocab: a set containing our x vocabulary
        #labels: a set containing our y vocabulary
        
        self.x = x_data
        self.y = y_data
        self.x_l = x_lemmas
        self.x_v = x_vocab
        self.l = labels
        self.l_d = label_dict
        
    

    def transition_funcs(self, labels):

#         transition_encodings = []

#         for y_prev in y_seq[:-1]:
#             for y in y_seq[1:]:
#                 transition_encodings += equivalence_func(x, i, x_seq) if y_prev == y_seq[i-1] and y == y_seq[i] else 0

#         return transition_encodings

         #return equivalence_func(x, i, x_seq) if y_prev == y_seq[i-1] and y == y_seq[i] else 0

    
    
        #the state function b(x, i) in Intro to CRFs Hanna M. Wallach et al.
        def equivalence_func(x, i, x_seq):

            return 1 if i < len(x_seq) and x == x_seq[i] else 0
        
        
        
        return [lambda y_prev, y_cur, i, x, y, x_l, y_prev_2 = yp2, y_cur_2 = yc2: equivalence_func(x[i] if i < len(x) else 'EOS', i, x) if y_prev == y_prev_2 and y_cur == y_cur_2 else 0
               for yp2 in labels for yc2 in labels]
    
        

    def state_funcs(self):

        features = []
        
        #feature 1: 1 if lemma form is same as base form, else 0
        features += [lambda y_prev, y_cur, i, x, y, x_l: 1 if i < len(x) and x_l is x[i] else 0]
        #feature 2: 1 if a numeral is present in x, else 0
        features += [lambda y_prev, y_cur, i, x, y, x_l: 1 if i < len(x) and re.search('[0-9]', x[i]) else 0]
        
        #feature 3: Laplace smoothing var. Can be adjusted
        features += [lambda y_prev, y_cur, i, x, y, x_l: 1]
        
        #feature 3: a "word/label state" feature: a feature that says "this word right here is a combination of this
        #word and this label."
#         features += [lambda y_prev, y_cur, i, x, y, x_l, x_word = x_word, y_label = y_label: 1 if y_label == y 
#                      and i < len(x) and x_word == x[i] else 0
#         for y_label in self.l
#         for x_word in self.x_v]

        return features

    #Generates features for a single sentence in out x dataset
    def featurize_sentence(self, sentence, labels, lemmas):
        funcs = self.transition_funcs(self.l) + self.state_funcs()
        feature_len = len(funcs)
        #Dimensions of our features: the amount of bigrams in our input sentence, all possible labels for n-1, all possible
        #labels for n, the amount of feature functions for each word
        features = np.zeros((len(sentence) + 1, len(self.l), len(self.l), feature_len))
        #print(features.shape)
        
        #Enumerates over all transition functions + state functions for all possible label sequences for each bigram. Added 1 to         #x range since we're counting bigrams
        for i in range(0, len(sentence) + 1):
            for j in range(0, len(self.l)):
                for k in range(0, len(self.l)):
                    for x in range(0, len(funcs)):
                        result = funcs[x](self.l[j], self.l[k], i, sentence, labels, lemmas)
                        features[i, j, k, x] = result
                        
                        
        return features

    
    def featurize_labels(self, labels):
        BOS = 'BOS'
        EOS = 'EOS'
        
        labels.insert(0, BOS)
        labels.append(EOS)
        
        #return [self.l_d[y] if y in self.l_d.keys() else self.l_d['OOV'] for y in labels]
        return [self.l_d[y] for y in labels]
    
    def generate_features(self):
        
        print("Now generating x...")
        x_vals = [self.featurize_sentence(x, y, xl) for x, y, xl in zip(self.x, self.y, self.x_l)]
        print("Now generating y...")
        y_vals = [self.featurize_labels(y) for y in self.y]
        
        return x_vals, y_vals
        


# In[ ]:




