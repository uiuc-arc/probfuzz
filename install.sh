sudo apt-get -y update
sudo apt-get install -y python2.7
sudo apt-get install -y python-pip
sudo apt-get install -y bc
sudo pip2 --no-cache-dir install antlr4-python2-runtime six astunparse ast pystan edward pyro-ppl==0.2.1 tensorflow==1.5.0 pandas
sudo pip2 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl 
(cd language/antlr/ && wget http://www.antlr.org/download/antlr-4.7.1-complete.jar && ./run.sh)
./check.py
