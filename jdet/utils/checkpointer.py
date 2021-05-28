import pickle

class Checkpointer:
    def __init__(self,model,optimizer,scheduler):
        self.model = model 
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def load(self,ckpt_file):
        ckpt = pickle.load(open(ckpt_file,"rb"))
        
    def save(self,ckpt_file):
        pickle.dump({},open(ckpt_file,"wb"))