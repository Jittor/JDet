import pickle
import torch 

class Checkpointer:
    def __init__(self,model=None,optimizer=None,scheduler=None):
        self.model = model 
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def load(self,ckpt_file):
        ckpt = pickle.load(open(ckpt_file,"rb"))
    
    def _load_torch(self,pth_file):
        state_dict = torch.load(pth_file)
        for k,d in state_dict.items():
            print()
            # print(k,d.keys())
        
    def save(self,ckpt_file):
        pickle.dump({},open(ckpt_file,"wb"))