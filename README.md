### ripenet

Neural architecture search w/ weight sharing.

Effort made to make this as close to [ENAS](https://arxiv.org/pdf/1802.03268.pdf) as possible, but had to guess at some of the details.

This is active research code -- so there's no documentation, and it's very unstable.  Open issues w/ questions and I'll help as I can.

#### Installation
```
conda create -n ripenet python=3.5 pip -y
source activate ripenet

conda install pytorch torchvision cuda90 -c pytorch -y
pip install -r requirements.txt

cd ~/projects/basenet
pip install -r requirements.txt
python setup.py clean --all install

cd ~/projects/ripenet
cd tests
python test.py

```

#### Dependencies

 - Python 3.6
 - Pytorch>=0.4 (installed from Github)
 - [basenet](https://github.com/bkj/basenet)