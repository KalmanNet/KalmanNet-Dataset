# KalmanNet-Dataset

## Parameters

* F/f: the evolution model for linear/non-linear cases
* H/h: the observation model for linear/non-linear cases
* q: evolution noise
* r: observation noise
* J: the order of Taylor series approximation 

## Linear case

For the synthetic linear dataset, we set F and H to take the controllable canonical and inverse canonical forms, respectively. F and H could take dimensions of 2x2, 5x5 and 10x10, while the evolution noise q and observation noise r take a constant gap of 20 dB. You could find sample datasets under Simulations/Linear_canonical/...

## Non-linear case

You could find Lorenz Attractor（LA) datasets under Simulations/Lorenz_Attractor/data/...

Inside this folder, data_gen.pt includes one trajectory of length 6,000,000 of LA model with J=5 and <img src="https://render.githubusercontent.com/render/math?math=\Delta t = 10^{-5}">. The other sub-folders include Discrete-Time datasets of LA model of different trajectory lengths T and with J=5.

## How to generate and load data

### linear case:

```
python main_linear.py
```
To change the parameters for your dataset, go to Extended_data.py


### non-linear case:

```
python main_lor.py
```
To change the parameters for your dataset, go to Simulations/Lorenz_Attractor/parameters.py

