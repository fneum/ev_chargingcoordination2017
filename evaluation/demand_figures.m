clc
clear all

datafiles = dir('../price_timeseries/*_15min*.txt');

filename = cell(length(datafiles),1);
data15 = cell(length(datafiles),1);

for k = 1:length(datafiles)
    filename{k} = datafiles(k).name;
    data15{k} = importdata(filename{k});
end

data215 = cell2mat(data15);
min15_ind = reshape(data215,[96,size(data215,1)/96]);
min15_agg = sum(min15_ind');

% ----------------------

datafiles = dir('../price_timeseries/*_5min*.txt');

filename = cell(length(datafiles),1);
data5 = cell(length(datafiles),1);

for k = 1:length(datafiles)
    filename{k} = datafiles(k).name;
    data5{k} = importdata(filename{k});
end

data25 = cell2mat(data5);
min5_ind = reshape(data25,[288,size(data25,1)/288]);
min5_agg = sum(min5_ind');

% ----------------------

datafiles = dir('../price_timeseries/*_1min*.txt');

filename = cell(length(datafiles),1);
data1 = cell(length(datafiles),1);

for k = 1:length(datafiles)
    filename{k} = datafiles(k).name;
    data1{k} = importdata(filename{k});
end

data21 = cell2mat(data1);
min1_ind = reshape(data21,[1440,size(data21,1)/1440]);
min1_agg = sum(min1_ind');

