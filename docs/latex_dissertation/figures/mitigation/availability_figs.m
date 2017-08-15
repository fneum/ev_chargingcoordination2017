x = 0:1:1440
cdf_dep = 1-normcdf(x,1285,130)
cdf_arr = normcdf(x,230,150)
pdf_dep = normpdf(x,1285,130)
pdf_arr = normpdf(x,230,150)

av_prob = min(cdf_dep,cdf_arr)

a = icdf('Normal',0.9,230,150)
b = icdf('Normal',0.1,1285,130)
c = icdf('Normal',0.5,230,150)
d = icdf('Normal',0.5,1285,130)

fig = figure;
hold on
xlim([0 1440])
set(fig,'defaultAxesColorOrder',[[0 0 0];[0 0 0]]);

fill([1 1 a a],[0 1 1 0], [.92 .92 .92],'EdgeColor','None')
fill([b b 1439 1439],[0 1 1 0], [.92 .92 .92],'EdgeColor','None')

fill([1 1 c c],[0 1 1 0], [.85 .85 .85],'EdgeColor','None')
fill([d d 1439 1439],[0 1 1 0], [.85 .85 .85],'EdgeColor','None')

plot(x,av_prob)
refline(0,0.5)
refline(0,0.9)

line([b,b],ylim)
line([a,a],ylim)



yyaxis right
ax = gca
ax.YColor = [0 0 0];
plot(x,pdf_dep,':k')
plot(x,pdf_arr,'--k')

refline(0,0)




