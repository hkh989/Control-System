%% Linearization
close all
clc

CW_real = [2.6969;
          3.0675]; % 왼쪽 모터 실제 dz and saturation in positive(cw)
        
CCW_real =  [2.345; 
            1.9739]; % 왼쪽 모터 실제 dz and saturation in negative(ccw)
         
matCoef_pos = [2.55,1;
                5,1];              % 1 dead zone, 360 saturation y축이 degree/sec임 
matCoef_neg = [2.45,1;
                0,1]  ;           % -1 dead zone, -360 saturation 
Coef_pos = matCoef_pos\CW_real; %왼쪽 모터 선형화 계수 in cw   

Coef_neg = matCoef_neg\CCW_real; %왼쪽 모터 선형화 계수 in ccw
 
Xc_pos1 = 2.45:0.05:5;
Xc_neg1 = 0:1.05:2.55;

Xcmd2motor_pos = Coef_pos(1)*Xc_pos1 +Coef_pos(2);
Xcmd2motor_neg = Coef_neg(1)*Xc_pos1 +Coef_neg(2);

%% Data Fitting
close all; clear all; clc;
data = load('Triangle0.97 2.480 V_data.csv');
time = data((1:16000),1);
Vcmd = data((1:16000),2);
Vgyro = data((1:16000),3);
omega = data((1:16000),4);


[b, a] = butter(2, 2*0.005*12, 'low');
LPResult(:,1)=filter(b, a, omega);%%output
[c,d] = butter(2,2*0.005*12,'low');
LPResult2(:,1)=filter(b,a,Vcmd);%%input
figure();
plot(time,LPResult2*120-300, 'r-'); hold on; grid on;
plot(time, LPResult, 'b-');
xlabel('Time[s]'), ylabel('\omega_n [deg/sec]'), title('TIme & \omega_n Sine Wave of 3[Hz]');
legend('Input Signal','Output Signal');
grid on;
%% BandPass Filter
% Variables
Freq        = 2.0000;
Wn          = [Freq - 0.5, Freq + 0.5];
sampleFreq  = 200;
sampleTime  = 0.005;
V_offset    = 2.5;
dummy = sprintf('SineFinal%.3f Hz_data.csv', Freq);
data  = load(dummy);
t     = data(:, 1);
Vc    = data(:, 2);
Vcmd  = data(:, 3);
omega = data(:, 4);

% 2nd BPF
[b, a] = butter(2, Wn / (sampleFreq / 2), 'bandpass');
LPFResult(:,1) = filter(b, a, omega);  % omega는 input

% MAIN
idealVc = Vc * 120 - 300;
LPF_Vc  = filter(b, a, idealVc);

% Plotting
plot(t, 120 * Vc - 300, 'Linewidth', 2), hold on
plot(t, LPF_Vc, 'Linewidth', 2), hold on
plot(t, omega, 'm', 'Linewidth', 1.2), hold on
plot(t, LPFResult, 'Linewidth', 2)

axis([2 4 -200 200]), grid on
xlabel('time [sec]'), ylabel('Magnitude [deg/sec]'), title('Phase shift of Sine Wave I/O (1Hz)', 'Fontsize', 15)
legend('sine input', 'sine input w/ BPF', 'sine output RAW', 'sine output w/ BPF')
%% Modeling Functions
 clear all ; close all ; clc ;

%f = [0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.80 0.85]
%f = 0.1
f = 0.5 : 0.1 : 1.7;

for i = 1 : length(f)
    dataName = sprintf('0603 Sin %.3f Hz_data.csv',f(i));
    File = load(dataName) ;
    
    DEG2RAD = pi/180 ;
    RAD2DEG = 180/pi ;

    time  = File((800:end),1) ;
    Vcmd  = File((800:end),2) * 120 -300; %[rad]
    Omega = File((800:end),4); %[deg/s]
    
    SysResp1 = inline('coef1(1)*sin(2*pi*fTime+coef1(2))+coef1(3)', 'coef1', 'fTime');
    SysResp2 = inline('coef2(1)*sin(2*pi*fTime+coef2(2))+coef2(3)', 'coef2', 'fTime');
    
    Vout0 = 2.5;
    
    coef1 = lsqcurvefit(@(coef1, time) coef1(1)*sin( 2*pi*f(i)*time + coef1(2) ) + coef1(3), [max(Vcmd) 0 (max(Vcmd)+min(Vcmd))/2], time, Vcmd);
    coef2 = lsqcurvefit(@(coef2, time) coef2(1)*sin( 2*pi*f(i)*time + coef2(2) ) + coef2(3), [max(Omega) 0 (max(Omega)+min(Omega))/2], time, Omega) ;
    
    eMag1   = coef1(1) ; % [V]   estimated magnitude
    ePhs1   = coef1(2) ; % [rad] estimated phase
    eBias1  = coef1(3) ; % [V]   estimated bias

    eMag2   = coef2(1) ; % [V]   estimated magnitude
    ePhs2   = coef2(2) ; % [rad] estimated phase
    eBias2  = coef2(3) ; % [V]   estimated bias

    
    eOmega1 = eMag1 * sin( 2*pi*f(i)*time + ePhs1 ); %+ eBias1 ; % estimated Vout
    eOmega2 = eMag2 * sin( 2*pi*f(i)*time + ePhs2 ) + eBias2 ; % estimated Vout
    
%     eVout1 = eMag1 * sin( 2*pi*f(i)*time + ePhs1 ) + eBias1 ; % estimated Vout
%     eVout2 = eMag2 * sin( 2*pi*f(i)*time + ePhs2 ) + eBias2 ; % estimated Vout

    Gain        = eMag2 / eMag1 ;
    Phase_Shift = (ePhs2 - ePhs1) * RAD2DEG; %[deg]
    
    tblOmega(i, 1)    = f(i) *(2*pi);
    tblMagAtt(i, 1)   = Gain ;
    tblPhsDelay(i, 1) = Phase_Shift * (DEG2RAD) ; %[rad]
end

figure(1)
plot(time, eOmega1, 'r','lineWidth', 0.5);
hold on;
xlabel('time');
ylabel('Omega[deg/s]');
plot(time, eOmega2, 'b', 'lineWidth', 0.5);
legend("Motor Modeling");1
hold off;
    
tblFreqResp = tblMagAtt .* exp(1i*tblPhsDelay) ;

Nnum = 0 ;
Nden = 1 ;

[num, den]   = invfreqs(tblFreqResp, tblOmega, Nnum, Nden)  
EstTransFunc = tf(num, den)

figure, bode(EstTransFunc)
grid on,set(gca, 'Fontsize', 13);

% [num, den]   = invfreqs(tblFreqResp, tblOmega, Nnum, Nden) ;
% EstTransFunc = tf(num, den) ;

[mag,phase,wout] = bode(EstTransFunc) ;
sysTF = tf([15.83], [1 20]);
[mag2,phase2,wout2] = bode(sysTF) ;

figure
subplot(2,1,1), semilogx(wout, mag2db(squeeze(mag)), 'linewidth', 2)
hold on, semilogx(tblOmega,  mag2db(tblMagAtt), 'ro', 'linewidth', 2)
grid on, title('Bode Diagram'), ylabel('Magnitude [dB]'), set(gca, 'Fontsize', 13)
subplot(2,1,2), semilogx(wout, squeeze(phase), 'linewidth', 2)
hold on, semilogx(tblOmega, tblPhsDelay * RAD2DEG, 'bo', 'linewidth', 2)
grid on, xlabel('Frequency [rad/s]'), ylabel('Phase [deg]'), set(gca, 'Fontsize', 13)

for i=1 : length(tblPhsDelay)
    phsdel(i)=tblPhsDelay(i)*180/pi;
    magatt(i)=tblMagAtt(i);
    
end






%% Margin Functions

 clear all ; close all ; clc ;

Zetac = [0.5; 0.6; 0.707; 0.80];
%Zetac=1/sqrt(2)
Pc = 0.1 : 0.01: 10;
tblPm = zeros(length(Pc),length(Zetac)) ;
x = zeros(length(Zetac), 1); y = zeros(length(Zetac), 1);


for j = 1 : length(Zetac)
    for i = 1 : length(Pc)

        Ki = Pc(i)^2 ;
        Kp = 2*Zetac(j)*Pc(i) - 1;

        numGc = [Kp Ki] ;
        denGc = [1  0 ] ;

        Gc = tf(numGc, denGc) ;

        numGm = [ 1 ] ;
        denGm = [ 1 1 ] ;

        Gm = tf(numGm, denGm) ;

        Go = Gc*Gm ;

        [Gm, Pm, Wcg, Wcp] = margin(Go) ;

        tblPm(i, j) = Pm ;

    end
end

figure, p = plot(Pc, tblPm, 'LineWidth', 2); grid on;
xlabel('\omega_c/\omega_m [-]'), ylabel('Phase Margin [deg]')
title('Phase Margin Evaluation')
legend(['\zeta_c = ' num2str(Zetac(1))], ['\zeta_c = ' num2str(Zetac(2))], ['\zeta_c = ' num2str(Zetac(3))], ['\zeta_c = ' num2str(Zetac(4))])

for i = 1 : length(Zetac)
    maximum = max(tblPm(:,j));
    [x(i), y(i)] = find(tblPm(:,j) == maximum);
    datatip(p(i), 'DataIndex', x(i));
end
arrTs = [1/sqrt(2), 0.65, 0.6];
Nts = length(arrTs);
% 
% for idxTs=1 : Nts
%     
%   Ts=arrTs(idxTs);
%   
%   strLgd{idxTs}=strcat('\zeta_c :', num2str(arrTs(idxTs)));
% end
% legend(strLgd);


% figure,
% plot(Pc, tblPm), grid on
% xlabel("\omega_c/\omega_m [-]"), ylabel("PM [deg]")
% title(['\zeta_c = ' num2str(Zetac) ])%(Pkp+Ki)/(P^2+(1+Kp)P+Ki
% where 1+kp refers to 2zetacpc, Ki=Pc^2
% Kp= 2zetacPc-1
% By iteration, we can approximate Km and corresponding Phase Margin
% Margin increased at the first moment of overshoot, then decrease as time
% goes. if the order of system is 1, it slowly decreases, while second
% order system drastically decreases its Phase Margin.

%% Phase Evaluation


clear all; close all; clc;

TauM = (1 / 18.53);
OmegaM = 1 / TauM;
ZetaM = 0.707;

OmegaC = 19 : 0.1 : 45;
Pc = OmegaC / OmegaM;

for i = 1:length(Pc)
    Ki = Pc(i)^2;
    Kp = 2 * ZetaM * Pc(i) -1;
    
    numGc = [Kp Ki];
    denGc = [1 0];
    
    Gc = tf(numGc, denGc);
    
    numGm = [1];
    denGm = [1 1];
    
    Gm = tf(numGm, denGm);
    
    Go = Gc * Gm;
    
    [GM PM(i)] = margin(Go);
    
end

figure(1)
plot(Pc, PM, 'r-')
hold on;
grid on;
title("Phase Margin \omega_c / \omega_m");
xlabel("(\omega_c / \omega_m)");
ylabel("PM[deg]");
hold off;
%% Data validation
% 13.1026886555386789<18.5<Wc<41.692500000000202
clear all,clc,close all;
Wc=23;%% 31

num=[13.63];
den=[1, 16.37];
transf=tf(num,den);

TauM=1/den(2);%% sec/rad
Km=num(1)/den(2);%% 
OmegaMax=6 %[rad/s]
Ki=Wc^2*TauM/Km;% Ki value
zetac=1/sqrt(2);%

Kp=(2*zetac*TauM*sqrt(Km*Ki/TauM)-1)/Km;
Error_Attenuation = abs((1i*OmegaMax*(1i*OmegaMax + 1/TauM)) / ((1i*OmegaMax)^2 + (1+Km*Kp)*(1i*OmegaMax)/TauM + (Km*Ki/TauM))) % Error<0.2
R = (1 / (2*TauM*zetac))
Limit = 2 / TauM

% N_d = sqrt((OmegaMax^4 * TauM^2 + OmegaMax^2))
% D_d = sqrt(a+b+c+d+e)
% Disturbance = N_d / D_d
numc=[1, 1/TauM, 0]
denc=[1, (1+Km*Kp)/TauM, Km*Ki/TauM]

trans_char=tf(numc,denc)


Gm = tf((Km/TauM), [1 1/TauM])
Gc = tf([Kp Ki], [1 0])
Go = Gm * Gc
Gcl = (Gc * Gm) / (1 + Gc * Gm)
Tr = (1-0.4167*zetac + 2.917*zetac^2) / Wc  %Rising Time < 0.2

[Gain,Phase,Wcg,Wcp] = margin(Go)
Gd = tf([1 1/TauM 0],[1 1/TauM*(1+Km*Kp) 1/TauM*Km*Ki]);%%c
circle(0,0,1/(2*TauM*zetac));
hold on;
circle(0,0,2.25/TauM);
circle(0,0,Wc);
syms Zt;
syms x;

EstZt=solve(exp(-1*(Zt*pi/sqrt(1-Zt^2)))==0.3,Zt,'Real',true);
Case4=sym2poly(EstZt);
x=-60:0.1:0;
y = tan((pi/2) - atan(Case4 / sqrt(1-Case4^2))) * x;
plot(x,y,'r--','LineWidth',2);
plot(x,-y,'r--','LineWidth',2);
plot(x,tan((pi/2)-atan(zetac/sqrt(1-zetac^2)))*x,'b--','LineWidth',2);
plot(x,-tan((pi/2)-atan(zetac/sqrt(1-zetac^2)))*x,'b--','LineWidth',2);
pzmap(Gcl);
grid on;
hold on;
xlim([-60 0]);
ylim([-60 60]);

pzmap(Gd)
pz = findobj(gca,'type','line');
set(pz(end),'MarkerSize',10,'MarkerEdgeColor','red','LineWidth',2);
set(pz(end-1),'MarkerSize',10,'MarkerEdgeColor','black','LineWidth',2);
set(pz(end-2),'MarkerSize',10,'MarkerEdgeColor','green','LineWidth',2);
legend('Overshoot','Zeta');


figure(2)
nyquist(Go)
ylim([-10 10])
 figure(3)
pzmap(Gm)

figure(5)
step(Gcl)
xlim([0 5])


%% Fitting
data = load('PI 2.761 V_data.csv');
cutoff_freq = 10;
sampleTime  = 0.005;
time = data((1:end),1);
response = data((1:end),4);
[b, a] = butter(2, 2*0.005*12, 'low');
LPResult(:,1)=filter(b, a, response);%%output
figure();
plot(time, LPResult, 'b-');
xlabel('Time[s]'), ylabel('\omega_n [deg/sec]'), title('TIme & \omega_n Sine Wave of 3[Hz]');
legend('Input Signal','Output Signal');
grid on;
%% 
plot(tspan,save_Wb);
hold on;
plot(tspan,save_Wg);
%%
Wc = 29; Zt_c = 1 / sqrt(2); % 32.8

Kp = (2 * Zt_c * Wc * TauM - 1) / Km;
Ki = TauM * Wc^2 / Km;

Gc = tf([Kp Ki], [1 0]);
Go = Gm * Gc;
Gcl = Go / (1 + Go);
[GM,PM,Wcg,Wcp] = margin(Go)

poleplacement = tf([1],[1 1/TauM*(1+Km*Kp) 1/TauM*Km*Ki]);
pzmap(poleplacement)
hold off

pz = findobj(gca,'type','line');
set(pz(end),'MarkerSize',10,'MarkerEdgeColor','red','LineWidth',2);
set(pz(end-2),'MarkerSize',10,'MarkerEdgeColor','black','LineWidth',2);
% set(pz(end-3),'MarkerSize',10,'MarkerEdgeColor','green','LineWidth',2);

Wd = tf([1 1/Tau_m 0], [1 1/Tau_m*(1+Km*Kp) 1/Tau_m*Km*Ki]);
figure, bodemag(Wd), grid on;

figure, nyquist(Go), grid on;
xlim([-2 1])
ylim([-1.5 1])

figure, bode(Gcl), grid on;
%% Kp Ki testing


for i = 1 : 30
Wc(i) = 18+i;
num=[13.63];
den=[1, 16.37];
transf=tf(num,den);
TauM=1/den(2);%% sec/rad
Km=num(1)/den(2);%% 
OmegaMax=6 %[rad/s]
Ki=Wc(i)^2*TauM/Km;% Ki value
zetac=1/sqrt(2);%
Kp=(2*zetac*TauM*sqrt(Km*Ki/TauM)-1)/Km;
Error_Attenuation = abs((1i*OmegaMax*(1i*OmegaMax + 1/TauM)) / ((1i*OmegaMax)^2 + (1+Km*Kp)*(1i*OmegaMax)/TauM + (Km*Ki/TauM))) % Error<0.2
R = (1 / (2*TauM*zetac))
Limit = 2 / TauM
ratio(i)= Ki/Kp;

end
plot(Wc,ratio,'r')

hold on
%% Linearization Testing
%%
 clear all; close all; clc;
addpath("C:\Users\USER\Desktop\학교\6학기\Digital Control and Signal Processing\Assignments\Final Project #1 Modeling\Data sets\2"); %파일이 저장되어 있는 주소

V = 0.0: 0.1 :5.0 ; % voltage 영역
a = zeros(length(V), 3);
V_offset = 1.301764;
omega_n = 0;
V_gimbal = 0;
omega = 0;
for i = 1 : length(V)
    dataName = sprintf('0522_Static_1 %.3 V_data.csv', V(i)); %csv data로 받음. 
    File     = load(dataName);
    
    Vcmd     = File(:,2);
    omega_n = File(:,4);
    
%     for j = 400 : 800
%         omega = omega + omega_n(j,1); %[V]
%     end
    omega = mean(omega_n(700:1200));
    
    
    %omega = (V_gimbal - V_offset) * 1000/0.67; %[deg/sec]
    
    a(i,1) = Vcmd(1,1);
    a(i,2) = omega;
    %d(i,1) = V_gimbal;
end

figure(2);
x = 0 : 0.001 : 5;
y = linspace(-300, 300, length(x));
plot(a(:,1), a(:,2), 'r-'); grid on; hold on;
% plot(x, y, 'r-');

xlabel('V_{cmd} [V]'), ylabel('\omega_n [deg/sec]'), title('V_{cmd} & \omega_n before Linearization');
legend('Before linearization','Ideal')
% ylim([-300 300]);
%%

close all; clear all; clc;
addpath("C:\Users\USER\Desktop\학교\6학기\Digital Control and Signal Processing\Assignments\Final Project #1 Modeling\Data sets\2"); %파일이 저장되어 있는 주소


V = 0 : 0.1 : 5.0 ; % voltage 영역
d = zeros(length(V), 3);

omega_n = 0;
V_gimbal = 0;
omega = 0;
for i = 1 : length(V)
    dataName = sprintf('0522_Static_1 %.3f V_data.csv', V(i)); %csv data로 받음. 
    File     = load(dataName);
    
    Vcmd     = File(:,2);
    omega_n = File(:,4);
    
%     for j = 700 : 1600
%         omega = omega + omega_n(j,1); %[V]
%     end
%     omega = omega/ 901;
    
    omega = mean(omega_n(600:end));
    %omega = (V_gimbal - V_offset) * 1000/0.67; %[deg/sec]
    
    d(i,1) = Vcmd(1,1);
    d(i,2) = omega;
    %d(i,1) = V_gimbal;
end

figure('DefaultAxesFontSize',15)
x = 0 : 0.001 : 5;
y = linspace(-300, 300, length(x));
plot(x, y, 'r-', 'LineWidth', 2); grid on; hold on;
plot(d(:,1), d(:,2), 'bo'); 


xlabel('V_{cmd} [V]'), ylabel('\omega_n [deg/sec]'), title('V_{cmd} & \omega_n After Linearization');
legend('V_{cmd}', 'After linearization')
%% Tracking

degree = -10 : 10 : 10;

for i = 1 : length(degree)
    dataName = sprintf('Tracking %.3f degree_data.csv', degree(i)); %csv data로 받음. 
    File     = load(dataName);
    DIR = File(:,5);
    DOA = File(:,6);
    DIRm= mean(DIR);
    DOAm = mean(DOA);
    
end

    plot(DIRm);
    hold on;

