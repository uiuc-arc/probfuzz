sudo apt-get update
sudo apt-get install python2.7
sudo apt-get install python-pip
sudo apt-get install bc
sudo pip2 --no-cache-dir install antlr4-python2-runtime six astunparse ast pystan edward pyro-ppl==0.1.2 tensorflow==1.5.0 pandas
sudo pip2 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl 
./check.py
