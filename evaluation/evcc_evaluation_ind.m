ind01 = csvread('../log/joint/iter1/sim/individual/simResults_household17.csv',1);
ind02 = csvread('../log/joint/iter1/sim/individual/simResults_household54.csv',1);
ind01_opt = csvread('../log/joint/iter1/opt/individual/optResults_household17.csv',1);
ind02_opt = csvread('../log/joint/iter1/opt/individual/optResults_household54.csv',1);

figure;
hold on
yyaxis left;
area(ind02(:,6))
plot(ind02(:,11))
refline(0,0.94)
yyaxis right;
plot(ind02(:,5))
plot(ind02_opt(:,5))
plot(ind02(:,9))
plot(ind02(:,13))


figure;
hold on
yyaxis left;
area(ind01(:,6))
plot(ind01(:,11))
refline(0,0.94)
yyaxis right;
plot(ind01(:,5))
plot(ind01_opt(:,5))
plot(ind01(:,9))
plot(ind01(:,13))


