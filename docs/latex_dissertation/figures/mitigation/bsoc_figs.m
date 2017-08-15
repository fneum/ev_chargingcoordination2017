y = 0:0.01:1;
x = 0:.1:9.5;

mu = (30-22*1.609*0.17) / 30 
sig = (30-15*1.609*0.17) / 30 /4

cdf = 1-normcdf(y,mu,sig);
pdf = normpdf(y,mu,sig);

mu2 = icdf('Normal',0.3,mu,sig)
mu3 = icdf('Normal',0.7,mu,sig)

fig = figure;
hold on
xlim([0 9.5])

fill([0.01 0.01 .7 .7],[0.001 .999 .999 0.001], [.92 .92 .92],'EdgeColor','None')
fill([0.01 0.01 .5 .5],[0.001 .999 .999 0.001], [.87 .87 .87],'EdgeColor','None')
fill([0.01 0.01 .3 .3],[0.001 .999 .999 0.001], [.82 .82 .82],'EdgeColor','None')
fill([5.4 5.4 6.2 6.2],[0.001 .999 .999 0.001], [.82 .82 .82],'EdgeColor','None')

line([1,1],ylim)

plot(1-cdf,y)

plot(pdf,y)


num1(1:96) = mu
num2(1:96) = mu2
num3(1:96) = mu3

for i = 56:63
    num1(i) = min(1,num1(i-1)+0.025075)
    num2(i) = min(1,num2(i-1)+0.025075)
    num3(i) = min(1,num3(i-1)+0.025075)
end

for i = 64:96
    num1(i) = max(num1)
    num2(i) = max(num2)
    num3(i) = max(num3)
end

plot(x,num1)
plot(x,num2)
plot(x,num3)


% 
% 
% av_prob = min(cdf_dep,cdf_arr)
% 
% a = icdf('Normal',0.9,230,150)
% b = icdf('Normal',0.1,1285,130)
% c = icdf('Normal',0.5,230,150)
% d = icdf('Normal',0.5,1285,130)
% 
% fig = figure;
% hold on

% set(fig,'defaultAxesColorOrder',[[0 0 0];[0 0 0]]);
% 

% fill([b b 1439 1439],[0 1 1 0], [.92 .92 .92],'EdgeColor','None')
% 
% fill([1 1 c c],[0 1 1 0], [.85 .85 .85],'EdgeColor','None')
% fill([d d 1439 1439],[0 1 1 0], [.85 .85 .85],'EdgeColor','None')
% 
% plot(x,av_prob)
% refline(0,0.5)
% refline(0,0.9)
% 
% line([b,b],ylim)
% line([a,a],ylim)
% 
% 
% 
% yyaxis right
% ax = gca
% ax.YColor = [0 0 0];
% plot(x,pdf_dep,':k')
% plot(x,pdf_arr,'--k')
% 
% refline(0,0)
% 
% 
% 
% 
