clear all
clc

% Defaults for this file
width = 3;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

set(0,'defaultLineLineWidth',1.5)

path = '../log/20170725-18_35_40/iter1';
rsa_opt = csvread('../log/uc/iter1/ref/refResults_SlotwiseAggregate.csv',1);
rsa_ref = csvread('../log/pg/iter1/ref/refResults_SlotwiseAggregate.csv',1);
rsa_sim = csvread('../log/lp_con/iter1/ref/refResults_SlotwiseAggregate.csv',1);
rsa_ben = csvread('../log/joint/iter1/ref/refResults_SlotwiseAggregate.csv',1);

figure;
subplot(3,3,1)
plot(sa_opt(:,5))
hold on;
plot(sa_sim(:,5))
plot(sa_ref(:,5))
plot(sa_ben(:,5))
hold off;
title('Schedule');
ylabel('kW');
xlim([0 96]);

subplot(3,3,2)
plot(sa_opt(:,6))
hold on;
plot(sa_sim(:,6))
plot(sa_ref(:,6))
plot(sa_ben(:,6))
plot(sa_ben(:,6))
hold off;
title('Availability');
ylabel('Number of EVs');
xlim([0 96]);

subplot(3,3,3)
plot(sa_opt(:,8))
hold on;
plot(sa_sim(:,8))
plot(sa_ref(:,8))
plot(sa_ben(:,8))
hold off;
title('Aggregate Battery SOC');
ylabel('[-]');
xlim([0 96]);


subplot(3,3,4)
plot(sa_opt(:,9))
hold on;
plot(sa_sim(:,9))
plot(sa_ref(:,9))
plot(sa_ben(:,9))
hold off;
title('Minimum Battery SOC');
ylabel('[-]');
xlim([0 96]);


subplot(3,3,5)
plot(sa_opt(:,11))
hold on;
plot(sa_sim(:,11))
plot(sa_ref(:,11))
plot(sa_ben(:,11))
hold off;
title('Minimum Voltage');
ylabel('pu');
xlim([0 96]);


subplot(3,3,6)
plot(sa_opt(:,12))
hold on;
plot(sa_sim(:,12))
plot(sa_ref(:,12))
plot(sa_ben(:,12))
hold off;
title('Line Loading');
ylabel('[-]');
xlim([0 96]);


subplot(3,3,7)
plot(sa_opt(:,13))
hold on;
plot(sa_sim(:,13))
plot(sa_ref(:,13))
plot(sa_ben(:,13))
plot(sa_ben(:,13))
hold off;
title('Electricity Price');
ylabel('p/kWh');
xlim([0 96]);

subplot(3,3,8)
plot(sa_opt(:,16))
hold on;
plot(sa_sim(:,16))
plot(sa_ref(:,16))
plot(sa_ben(:,16))
hold off;
title('Charging Cost');
ylabel('\pound');
xlim([0 96]);

subplot(3,3,9)
plot(sa_opt(:,3))
hold on;
plot(sa_sim(:,3))
plot(sa_ref(:,3))
plot(sa_ben(:,3))
plot(sa_ben(:,3))
hold off;
title('Residential Load');
ylabel('kW');
xlim([0 96]);

% mc_afap = csvread('../log/20170717-13_19_11/Results_MonteCarloDistributions.csv',1);
% mc_price = csvread('../log/20170717-12_07_10/Results_MonteCarloDistributions.csv',1);
% mc_lp = csvread('../log/20170717-11_51_04/Results_MonteCarloDistributions.csv',1);
% 
% figure;
% bar([mc_lp(:,2),mc_lp(:,9),mc_lp(:,16)]);
% 
% 
% path = '../log/20170717-12_07_10/iter14';
% path2 = '../log/20170717-11_51_04/iter14';
% test = csvread(strcat(path,'/sim/simResults_EVavailability.csv'));
% test2 = csvread(strcat(path2,'/sim/simResults_EVavailability.csv'));
% figure;
% subplot(1,2,1);
% heatmap(test);
% subplot(1,2,2);
% heatmap(test2);
