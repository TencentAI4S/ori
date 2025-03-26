# Install Tutorial

## Miniconda
```shell
wget -O minicnda3.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_24.5.0-0-Linux-x86_64.sh
bash minicnda3.sh -b -p /miniconda
rm minicnda3.sh
conda init
conda env create -n ori -f environment.yaml
conda activate ori
```

## Pip
```shell
pip install --no-cache-dir -r requirements.txt
```

## setup.py
```shell
python setup.py install
```

## Other Options

### Flash Attention
flash attention speed up for multihead attention
```shell
MAX_JOBS=4 pip install -U --no-cache-dir flash-attn==2.5.2 --no-build-isolation
```

### Apex
apex speed up for rmsnorm and layernorm

```shell
git clone https://github.com/NVIDIA/apex  \
&& cd apex  \
&& git checkout 24.04.01  \
&& pip install -r requirements.txt  \
&& python3 setup.py install --cpp_ext --cuda_ext  \
&& cd ..
```