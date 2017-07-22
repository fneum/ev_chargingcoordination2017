data = csvread('data/hourlyload_uk_2016_corrected.csv');
dataW = csvread('data/hourlyload_uk_2016_winter.csv');
dataS = csvread('data/hourlyload_uk_2016_summer.csv');
dataT = csvread('data/hourlyload_uk_2016_transition.csv');

subplot(2,2,1);
dailydata = reshape(data,[24,size(data,1)/24]);
Z = dailydata.';
boxplot(Z)

subplot(2,2,2);
dailydataW = reshape(dataW,[24,size(dataW,1)/24]);
ZW = dailydataW.';
boxplot(ZW)

subplot(2,2,3);
dailydataT = reshape(dataT,[24,size(dataT,1)/24]);
ZT = dailydataT.';
boxplot(ZT)

subplot(2,2,4);
dailydataS = reshape(dataS,[24,size(dataS,1)/24]);
ZS = dailydataS.';
boxplot(ZS)

%Z = resizem(newdata,[269 96]);
%xaxis = 1:1440;
%B = repmat(xaxis,1,size(data,1)/1440).'