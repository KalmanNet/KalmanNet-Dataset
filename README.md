# KalmanNet-Dataset

## Parameters

* F/f: the evolution model for linear/non-linear cases
* H/h: the observation model for linear/non-linear cases
* q: evolution noise
* r: observation noise

## Linear case

For the synthetic linear dataset, we set F and H to take the controllable canonical and inverse canonical forms, respectively. F and H could take dimensions of 2x2, 5x5 and 10x10, while the evolution noise q and observation noise r take a constant gap of 20 dB.

## Non-linear case

data_gen.pt includes one trajectory of length 6,000,000 of Lorenz Attractor model with J=5 and - <img src="https://latex.codecogs.com/png.latex?\Delta t = 10^{-5}" />. 

