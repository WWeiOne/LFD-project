# LSTNet
Refenrence: "Modeling long-and short-term temporal patterns with deep neural networks." 

https://github.com/laiguokun/LSTNet


## Usage
1. Download electricity dataset from 

https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
2. Unzip and place under ./data/
3. Pre-process
```
python load_electricity.py
```

3. Train
```
python main.py --gpu 0 --save save/elec.pt --output_fun Linear
```



