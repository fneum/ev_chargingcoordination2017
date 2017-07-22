clc
clear all

datafiles = dir('../price_timeseries/*.txt');

filename = cell(length(datafiles),1);
data30 = cell(length(datafiles),1);

for k = 1:length(datafiles)
    filename{k} = datafiles(k).name;
    data30{k} = importdata(filename{k});
end

all = cell2mat(data30);
individual = reshape(all,[48,size(all,1)/48]);

all_spread = (all-mean(all)).*5+mean(all)

