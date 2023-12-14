# LSTNet
Refenrence: "Modeling long-and short-term temporal patterns with deep neural networks." 

https://github.com/laiguokun/LSTNet


## Datasets
Download from 

https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

Unzip and place under ./data/

## Pre-process

```
python load_electricity.py
```

## How to run it

```
python main.py --gpu 0 --horizon 24 --save save/elec.pt --output_fun Linear
```

