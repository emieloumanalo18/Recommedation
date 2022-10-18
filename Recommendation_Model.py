import numpy as np
import re
from collections import defaultdict



settings ={
    'n_dimesnion': 4,
    'epochs' :40,
    'learning_rate' : 0.01
}

         

class NueralNetwork():
    def __init__ (self):
        self.n = settings['n_dimesnion']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        pass
        
        
    def generate_training_data(self, data_pairs):
        count_x = defaultdict(int)
        count_y = defaultdict(int)
        
        for i in range(len(data_pairs)):
            count_x[data_pairs[i][0]] +=1
            count_y[data_pairs[i][1]] +=1

        self.len_x = len(count_x.keys())
        self.len_y = len(count_y.keys())
        
        self.list_x = sorted(list(count_x.keys()), reverse=False)
        self.x_index = {x:i for (i, x) in enumerate(self.list_x)}
        self.index_x = {i:x for (i, x) in enumerate(self.list_x)}
        
        self.list_y = sorted(list(count_y.keys()), reverse=False)
        self.y_index = {x:i for (i, x) in enumerate(self.list_y)}
        self.index_y = {i:x for (i, x) in enumerate(self.list_y)}
        
        
        
        
        for i in range(len(data_pairs)):
            training_data = []
            for c, (x, y) in enumerate(data_pairs):
                target = self.x_index[x] 
                w_target = [0 for i in range(0, self.len_x )]
                w_target[target] = 1


                content = self.y_index[y]
                w_content = [0 for i in range(0, self.len_y )]
                w_content[content] = 1

                training_data.append([w_target, w_content])

        return np.array(training_data, dtype=object)
    
    
    
    
    def batch(self, data_pairs):
        bt = np.random.choice(np.shape(data_pairs)[0], size=100)

        train_data =[]
        for i in bt:
            train_data.append(data_pairs[i])

        yield train_data  



    def tokenizer(self, words):
        token = []
        for word in words:
            token.append(word.split())

        return token 



    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    

        

    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u
                

        
        
    def backprop(self, e, h, x): 
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        
       
    
    
    def train(self, training_data):
        self.w1 = np.random.uniform(-1, 1, (self.len_x, self.n))    
        self.w2 = np.random.uniform(-1, 1, (self.n, self.len_x))   
             
                  
        for i in range(self.epochs):
            self.loss = 0
            train_batch = next(self.batch(training_data))
            
            for c, (w_t, w_c) in enumerate(train_batch):
                y_pred, h, u = self.forward_pass(w_t)
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                self.backprop(EI, h, w_t)
                
                sum_u = np.logaddexp(0, np.sum(u))  
                self.loss += -np.sum(u[w_c.index(1)]) + len(w_c) * sum_u

            print('Iteration: ',i, ' Loss: ', self.loss)
        
    
    
    
    

    def word_vec(self, word):
        w_index = self.x_index[word]
        v_w = self.w1[w_index]
        return v_w
    
    
    
    
    def vec_sim(self, word, top_n=10):
        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.len_x):
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den
                
            word = self.index_x[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
        
        get_word = []
        get_sim = []
        
        for word, sim in words_sorted[:top_n]:
            get_word.append([word])
#             get_sim.append(sim)
            
        return get_word
 
     

nn = NueralNetwork()



