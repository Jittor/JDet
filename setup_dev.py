# add jdet to $PYTHONPATH for development temporarily.
import os
jdet_path = os.path.join(os.getcwd(), 'python')
cmd = 'export PYTHONPATH=$PYTHONPATH:'+jdet_path
print(cmd)
with open(os.path.join(os.environ['HOME'], ".bashrc"), mode="a") as file:  
    file.write(cmd)  
with open(os.path.join(os.environ['HOME'], ".zshrc"), mode="a") as file:  
    file.write(cmd)  