# LSTNet
Refenrence: "Modeling long-and short-term temporal patterns with deep neural networks." 

https://github.com/laiguokun/LSTNet

## Modifications
* write load_electricity.py to pre-process raw dataset
* write ./main.py/get_data() function to split data into splices
* write ./main.py/get_data_test() function to generate a linear signal with noise for comparison
* rewrite ./main.py/evaluation function to plot truth-pred comparison every 5 epoch
* write ./outputs/Plot/imshow_all.py to plot training, testing, prediction in one figure

## Usage
1. Download electricity dataset from 

https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

2. Unzip and place under ./data/
  
4. Pre-process

```
python load_electricity.py
```

3. Train & test

```
python main.py --gpu 0 --save save/elec.pt --output_fun Linear
```

3. Train & test with generated linear signal with noise 

```
python main.py --gpu 0 --save save/elec.pt --output_fun Linear --high_window 0 --egsignal True
```

4. Results would be saved under ./outputs/ and can be plotted by ./outputs/imshow_all.py



