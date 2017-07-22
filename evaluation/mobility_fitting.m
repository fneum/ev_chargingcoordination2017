clear all
clc

load tripend_sigmu
load tripstart_sigmu
load mileage_sigmu

% RUN FITS
fit_te_sigma = allfitdist(tripend_sigmu(:,1));
fit_te_mu = allfitdist(tripend_sigmu(:,2));
fit_ts_sigma = allfitdist(tripstart_sigmu(:,1));
fit_ts_mu = allfitdist(tripstart_sigmu(:,2));
fit2_ts_sigma = fitdist(tripstart_sigmu(:,1),'HalfNormal');
fit_mi_mu = allfitdist(mileage_sigmu(:,2));
fit_mi_sigma = allfitdist(mileage_sigmu(:,1));

% PRINT SELECTED PARAMETERS
fit_ts_mu(1);
fit_ts_sigma(2);
fit_te_mu(1);
fit_te_sigma(3);
fit_mi_mu(1);
fit_mi_sigma(2);

% PLOT HISTOGRAMS AND FITTED PDFS
subplot(3,2,1)
ts_a = histfit(tripstart_sigmu(:,2),96,'gamma');
title('TripStart Average Distribution Fit');
ylabel('Frequency');
xlabel('Time [Minutes after Midnight]');
xlim([0 1440]);
ts_a(1).FaceColor = [.8 .8 .8];
ts_a(2).Color = [.2 .2 .2];
dim = [0.35 0.9 0 0];
str1 = {fit_ts_mu(6).DistName,
    string(fit_ts_mu(6).ParamNames(1)) + " = " + string(fit_ts_mu(6).Params(1)),
    string(fit_ts_mu(6).ParamNames(2)) + " = " + string(fit_ts_mu(6).Params(2))};
annotation('textbox',dim,'String',str1,'FitBoxToText','on');

subplot(3,2,2)
% ts_s = histfit(tripstart_sigmu(:,1),50,'generalized extreme value');
ts_s = histfit(tripstart_sigmu(:,1),50,'HalfNormal');
title('TripStart Deviation Distribution Fit');
ylabel('Frequency');
xlabel('Standard Deviation [Minutes from Mean]');
xlim([0 800]);
ts_s(1).FaceColor = [.8 .8 .8];
ts_s(2).Color = [.2 .2 .2];
dim = [0.8 0.9 0 0];
% str2 = { fit_ts_sigma(2).DistName,
%     string(fit_ts_sigma(2).ParamNames(1)) + " = " + string(fit_ts_sigma(2).Params(1)),
%     string(fit_ts_sigma(2).ParamNames(2)) + " = " + string(fit_ts_sigma(2).Params(2)),
%     string(fit_ts_sigma(2).ParamNames(3)) + " = " + string(fit_ts_sigma(2).Params(3))};
str2 = { "HalfNormal", "mu = " + fit2_ts_sigma.mu, "sigma = " + fit2_ts_sigma.sigma};
annotation('textbox',dim,'String',str2,'FitBoxToText','on');

subplot(3,2,3)
te_a = histfit(tripend_sigmu(:,2),96,'logistic');
title('TripEnd Average Distribution Fit');
ylabel('Frequency');
xlabel('Time [Minutes after Midnight]');
xlim([0 1440]);
te_a(1).FaceColor = [.8 .8 .8];
te_a(2).Color = [.2 .2 .2];
dim = [0.2 0.6 0 0];
str3 = { fit_te_mu(2).DistName,
    string(fit_te_mu(2).ParamNames(1)) + " = " + string(fit_te_mu(2).Params(1)),
    string(fit_te_mu(2).ParamNames(2)) + " = " + string(fit_te_mu(2).Params(2))};
annotation('textbox',dim,'String',str3,'FitBoxToText','on');

subplot(3,2,4)
te_s = histfit(tripend_sigmu(:,1),50,'logistic');
title('TripEnd Deviation Distribution Fit');
ylabel('Frequency');
xlabel('Standard Deviation [Minutes from Mean]');
xlim([0 800]);
te_s(1).FaceColor = [.8 .8 .8];
te_s(2).Color = [.2 .2 .2];
dim = [0.8 0.6 0 0];
str4 = { fit_te_sigma(3).DistName,
    string(fit_te_sigma(3).ParamNames(1)) + " = " + string(fit_te_sigma(3).Params(1)),
    string(fit_te_sigma(3).ParamNames(2)) + " = " + string(fit_te_sigma(3).Params(2))};
annotation('textbox',dim,'String',str4,'FitBoxToText','on');

subplot(3,2,5)
te_s = histfit(mileage_sigmu(:,1),96,'gamma');
title('Mileage Average Distribution Fit');
ylabel('Frequency');
xlabel('Time [Minutes after Midnight]');
xlim([0 100]);
te_s(1).FaceColor = [.8 .8 .8];
te_s(2).Color = [.2 .2 .2];
dim = [0.35 0.3 0 0];
str5 = { fit_mi_mu(1).DistName,
    string(fit_mi_mu(1).ParamNames(1)) + " = " + string(fit_mi_mu(1).Params(1)),
    string(fit_mi_mu(1).ParamNames(2)) + " = " + string(fit_mi_mu(1).Params(2))};
annotation('textbox',dim,'String',str5,'FitBoxToText','on');

subplot(3,2,6)
te_s = histfit(mileage_sigmu(:,1),50,'exponential');
title('Mileage Deviation Distribution Fit');
ylabel('Frequency');
xlabel('Standard Deviation [Minutes from Mean]');
te_s(1).FaceColor = [.8 .8 .8];
te_s(2).Color = [.2 .2 .2];
dim = [0.8 0.3 0 0];
str6 = { fit_mi_sigma(2).DistName,
    string(fit_mi_sigma(2).ParamNames(1)) + " = " + string(fit_mi_sigma(2).Params(1))};
annotation('textbox',dim,'String',str6,'FitBoxToText','on');

% GENERATE RANDOM SAMPLES FROM FITTED DISTRIBUTION FUNCTION

% TripStart Average Distribution
pd_tsm = makedist('Loglogistic','mu',fit_ts_mu(1).Params(1),'sigma',fit_ts_mu(1).Params(2));
random_tsm = random(pd_tsm,4000,1);

% TripStart SD Distributioin
pd_tss = makedist('GeneralizedExtremeValue','k',fit_ts_sigma(2).Params(1),'sigma',fit_ts_sigma(2).Params(2),'mu',fit_ts_sigma(2).Params(3));
pd_tss_trunc = truncate(pd_tss,0,inf);
random_tss = random(pd_tss_trunc,4000,1);

% TripEnd Average Distribution
pd_tem = makedist('tlocationscale','mu',fit_te_mu(1).Params(1),'sigma',fit_te_mu(1).Params(2),'nu',fit_te_mu(1).Params(3));
random_tem = random(pd_tem,4000,1);

% TripEnd SD Distribution
pd_tes = makedist('logistic','mu',fit_te_sigma(3).Params(1),'sigma',fit_te_sigma(3).Params(2));
pd_tes_trunc = truncate(pd_tes,0,inf);
random_tes = random(pd_tes_trunc,4000,1);

% Dependent Simulation of TripStart Average and SD Distributions
cc_ts_empirical = corrcoef(tripstart_sigmu(:,1),tripstart_sigmu(:,2))
cov_ts_empirical = cov(tripstart_sigmu(:,1),tripstart_sigmu(:,2))
Z = mvnrnd([0 0], cc_ts_empirical, 2000);
U = normcdf(Z);
% X = [ gevinv(U(:,1),fit_ts_sigma(2).Params(1),fit_ts_sigma(2).Params(2),fit_ts_sigma(2).Params(3)) icdf('LogLogistic',U(:,2),fit_ts_mu(1).Params(1),fit_ts_mu(1).Params(2))];
random_ts_sigmu =  [ icdf('HalfNormal',U(:,1),fit2_ts_sigma.mu,fit2_ts_sigma.sigma) icdf('LogLogistic',U(:,2),fit_ts_mu(1).Params(1),fit_ts_mu(1).Params(2))];

cc_ts_simulated = corrcoef(random_ts_sigmu(:,1),random_ts_sigmu(:,2))

figure;
scatterhist(random_ts_sigmu(:,1),random_ts_sigmu(:,2))
xlim([0 600]);
ylim([0 1440]);
title('Simulated TripStart Distribution');
ylabel('\mu');
xlabel('\sigma');
set(get(gca,'children'),'marker','.')

figure;
scatterhist(tripstart_sigmu(:,1),tripstart_sigmu(:,2))
xlim([0 600]);
ylim([0 1440]);
title('Empirical TripStart Distribution');
ylabel('\mu');
xlabel('\sigma');
set(get(gca,'children'),'marker','.')

% Dependent Simulation of Mileage Average and SD Distributions

cc_mi_empirical = corrcoef(mileage_sigmu(:,1),mileage_sigmu(:,2))
cov_mi_empirical = cov(mileage_sigmu(:,1),mileage_sigmu(:,2))
Z = mvnrnd([0 0], cc_mi_empirical, 500);
% Z = mvnrnd([0 0], [1 0.75;0.75 1], 500);

U = normcdf(Z);
random_mi_sigmu =  [ icdf('Exponential',U(:,1),fit_mi_sigma(2).Params(1)) icdf('Gamma',U(:,2),fit_mi_mu(1).Params(1),fit_mi_mu(1).Params(2)) ];
% random_mi_sigmu =  [ icdf('GeneralizedExtremeValue',U(:,1),fit_mi_sigma(3).Params(1),fit_mi_sigma(3).Params(2),fit_mi_sigma(3).Params(3)) icdf('Gamma',U(:,2),fit_mi_mu(1).Params(1),fit_mi_mu(1).Params(2)) ];

cc_mi_simulated = corrcoef(random_mi_sigmu(:,1),random_mi_sigmu(:,2))

figure;
scatterhist(random_mi_sigmu(:,1),random_mi_sigmu(:,2))
xlim([0 100]);
ylim([0 100]);
title('Simulated Mileage Distribution');
ylabel('\mu');
xlabel('\sigma');
set(get(gca,'children'),'marker','.')

figure;
scatterhist(mileage_sigmu(:,1),mileage_sigmu(:,2))
xlim([0 100]);
ylim([0 100]);
title('Empirical Mileage Distribution');
ylabel('\mu');
xlabel('\sigma');
set(get(gca,'children'),'marker','.')
