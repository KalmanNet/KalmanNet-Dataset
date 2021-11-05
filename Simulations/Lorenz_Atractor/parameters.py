import torch
import math

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

#########################
### Design Parameters ###
#########################
m = 3
n = 3
variance = 0
m1x_0 = torch.ones(m, 1) 
m1x_0_design_test = torch.ones(m, 1)
m2x_0 = 0 * 0 * torch.eye(m)

# Noise Parameters
r2 = torch.tensor([1])
# r2 = torch.tensor([100, 10, 1, 0.1, 0.01])
r = torch.sqrt(r2)
vdB = -20 # ratio v=q2/r2 in dB
v = 10**(vdB/10)
q2 = torch.mul(v,r2)
q = torch.sqrt(q2)

# Taylor expansion order
J = 5
J_mod = 2

# Decimation ratio
delta_t_gen =  1e-5
delta_t = 0.02
delta_t_mod = 0.02
delta_t_test = 0.01
ratio = delta_t_gen/delta_t_test

# Length of Time Series Sequence
# T = math.ceil(3000 / ratio)
# T_test = math.ceil(6e6 * ratio)
T = 100
T_test = 100

#################################################
### Generative Parameters For Lorenz Atractor ###
#################################################

# Auxiliar MultiDimensional Tensor B and C (they make A --> Differential equation matrix)
B = torch.tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(m,m), torch.zeros(m,m)]).float()
C = torch.tensor([[-10, 10,    0],
                  [ 28, -1,    0],
                  [  0,  0, -8/3]]).float()


H_design = torch.eye(3)

## Angle of rotation in the 3 axes
roll_deg = yaw_deg = pitch_deg = 1

roll = roll_deg * (math.pi/180)
yaw = yaw_deg * (math.pi/180)
pitch = pitch_deg * (math.pi/180)

RX = torch.tensor([
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]])
RY = torch.tensor([
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]])
RZ = torch.tensor([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]])

RotMatrix = torch.mm(torch.mm(RZ, RY), RX)
H_mod = torch.mm(RotMatrix,H_design)


H_design_inv = torch.inverse(H_design)

# Noise Parameters
Q_non_diag = False
R_non_diag = False

Q = (q**2) * torch.eye(m)

if(Q_non_diag):
    q_d = q**2
    q_nd = (q **2)/2
    Q = torch.tensor([[q_d, q_nd, q_nd],[q_nd, q_d, q_nd],[q_nd, q_nd, q_d]])

R = (r**2) * torch.eye(n)

if(R_non_diag):
    r_d = r**2
    r_nd = (r **2)/2
    R = torch.tensor([[r_d, r_nd, r_nd],[r_nd, r_d, r_nd],[r_nd, r_nd, r_d]])

##############################################
#### Model Parameters For Lorenz Atractor ####
##############################################

# Auxiliar MultiDimensional Tensor B and C (they make A)
B_mod = torch.tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(m,m), torch.zeros(m,m)])
C_mod = torch.tensor([[-10, 10,    0],
                      [ 28, -1,    0],
                      [  0,  0, -8/3]])

H_mod_inv = torch.inverse(H_mod)