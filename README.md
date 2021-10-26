# KalmanNet-Dataset

## Parameters

* F/f: the evolution model for linear/non-linear cases
* H/h: the observation model for linear/non-linear cases
* q: evolution noise
* r: observation noise
* J: the order of Taylor series approximation 

## Linear case

For the synthetic linear dataset, we set F and H to take the controllable canonical and inverse canonical forms, respectively. F and H could take dimensions of 2x2, 5x5 and 10x10, while the evolution noise q and observation noise r take a constant gap of 20 dB.

## Non-linear case

data_gen.pt includes one trajectory of length 6,000,000 of Lorenz Attractor（LA) model with J=5 and <img src="https://render.githubusercontent.com/render/math?math=\Delta t = 10^{-5}">. The other folders includes Discrete-Time datasets of LA model of different trajectory lengths and with J=5.

