clc;
clear all;
close all;
%% Matrices B and E
Bi = [0;1];
B0 = zeros(2,1);

B = [-Bi,B0,B0,Bi,-Bi,B0,B0,B0,B0,B0,B0;
      Bi,Bi,Bi,B0,B0,Bi,-Bi,B0,B0,B0,B0;
      B0,B0,-Bi,B0,B0,B0,B0,Bi,B0,B0,B0;
      B0,B0,B0,-Bi,B0,B0,B0,B0,Bi,B0,Bi;
      B0,B0,B0,B0,Bi,-Bi,B0,B0,-Bi,Bi,B0;
      B0,B0,B0,B0,B0,B0,Bi,-Bi,B0,-Bi,B0;
    ];

Ei = [0;1];
E0 = zeros(2,1);
E = [Ei;E0;Ei;E0;Ei;Ei];

Ai = [0,1;
      0,0];
A = blkdiag(Ai,Ai,Ai,Ai,Ai,Ai);
sigma= 0.15;
A_ = A + sigma*eye(12);
%% Feedback control
setlmis([])

%variables to compute
S = lmivar(1,[12 1]);
gamma = lmivar(1,[1 0]);

%matrix inequalities to ensure stability of the marginally stable system
lmiterm([1 1 1 S],1,A_,'s');
lmiterm([1 1 1 gamma],1,-2,B*B');

% S and gamma grater tha zero
lmiterm([-2 1 1 S],1,1);
lmiterm([-3 1 1 gamma],1,1);

lmis = getlmis;
[tmin,xfeas] = feasp(lmis);

S_sol = dec2mat(lmis,xfeas,S);
gamma_sol = dec2mat(lmis,xfeas,gamma);

P = S_sol^-1;
K = gamma_sol * B'*P;

%% Simulation
time =[0,150];
x0 = zeros(1,12);

[t,x] = ode45(@(t,x) A*x-B*K*x-E,time,x0);

%% Plotting
figure();
hold on; grid on;
plot(t,x);
xlabel("time");
ylabel("Value of states");
title("Simulation of distribuited control system");


