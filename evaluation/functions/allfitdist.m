function [D PD] = allfitdist(data,sortby,varargin)
%ALLFITDIST Fit all valid parametric probability distributions to data.
%   [D PD] = ALLFITDIST(DATA) fits all valid parametric probability
%   distributions to the data in vector DATA, and returns a struct D of
%   fitted distributions and parameters and a struct of objects PD
%   representing the fitted distributions. PD is an object in a class
%   derived from the ProbDist class.
%
%   [...] = ALLFITDIST(DATA,SORTBY) returns the struct of valid distributions
%   sorted by the parameter SORTBY
%        NLogL - Negative of the log likelihood
%        BIC - Bayesian information criterion (default)
%        AIC - Akaike information criterion
%        AICc - AIC with a correction for finite sample sizes
%
%   [...] = ALLFITDIST(...,'DISCRETE') specifies it is a discrete
%   distribution and does not attempt to fit a continuous distribution
%   to the data
%
%   [...] = ALLFITDIST(...,'PDF') or (...,'CDF') plots either the PDF or CDF
%   of a subset of the fitted distribution. The distributions are plotted in
%   order of fit, according to SORTBY.
%
%   List of distributions it will try to fit
%     Continuous (default)
%       Beta
%       Birnbaum-Saunders
%       Exponential
%       Extreme value
%       Gamma
%       Generalized extreme value
%       Generalized Pareto
%       Inverse Gaussian
%       Logistic
%       Log-logistic
%       Lognormal
%       Nakagami
%       Normal
%       Rayleigh
%       Rician
%       t location-scale
%       Weibull
%
%     Discrete ('DISCRETE')
%       Binomial
%       Negative binomial
%       Poisson
%
%   Optional inputs:
%   [...] = ALLFITDIST(...,'n',N,...)
%   For the 'binomial' distribution only:
%      'n'            A positive integer specifying the N parameter (number
%                     of trials).  Not allowed for other distributions. If
%                     'n' is not given it is estimate by Method of Moments.
%                     If the estimated 'n' is negative then the maximum
%                     value of data will be used as the estimated value.
%   [...] = ALLFITDIST(...,'theta',THETA,...)
%   For the 'generalized pareto' distribution only:
%      'theta'        The value of the THETA (threshold) parameter for
%                     the generalized Pareto distribution. Not allowed for
%                     other distributions. If 'theta' is not given it is
%                     estimated by the minimum value of the data.
%
%   Note: ALLFITDIST does not handle nonparametric kernel-smoothing,
%   use FITDIST directly instead.
%
%
%   EXAMPLE 1
%     Given random data from an unknown continuous distribution, find the
%     best distribution which fits that data, and plot the PDFs to compare
%     graphically.
%        data = normrnd(5,3,1e4,1);         %Assumed from unknown distribution
%        [D PD] = allfitdist(data,'PDF');   %Compute and plot results
%        D(1)                               %Show output from best fit
%
%   EXAMPLE 2
%     Given random data from a discrete unknown distribution, with frequency
%     data, find the best discrete distribution which would fit that data,
%     sorted by 'NLogL', and plot the PDFs to compare graphically.
%        data = nbinrnd(20,.3,1e4,1);
%        values=unique(data); freq=histc(data,values);
%        [D PD] = allfitdist(values,'NLogL','frequency',freq,'PDF','DISCRETE');
%        PD{1}
%
%  EXAMPLE 3
%     Although the Geometric Distribution is not listed, it is a special
%     case of fitting the more general Negative Binomial Distribution. The
%     parameter 'r' should be close to 1. Show by example.
%        data=geornd(.7,1e4,1); %Random from Geometric
%        [D PD]= allfitdist(data,'PDF','DISCRETE');
%        PD{1}
%
%  EXAMPLE 4
%     Compare the resulting distributions under two different assumptions
%     of discrete data. The first, that it is known to be derived from a
%     Binomial Distribution with known 'n'. The second, that it may be
%     Binomial but 'n' is unknown and should be estimated. Note the second
%     scenario may not yield a Binomial Distribution as the best fit, if
%     'n' is estimated incorrectly. (Best to run example a couple times
%     to see effect)
%        data = binornd(10,.3,1e2,1);
%        [D1 PD1] = allfitdist(data,'n',10,'DISCRETE','PDF'); %Force binomial
%        [D2 PD2] = allfitdist(data,'DISCRETE','PDF');       %May be binomial
%        PD1{1}, PD2{1}                             %Compare distributions
%

%    Mike Sheppard
%    Last Modified: 17-Feb-2012




%% Check Inputs
if nargin == 0
    data = 10.^((normrnd(2,10,1e4,1))/10);
    sortby='BIC';
    varargin={'CDF'};
end
if nargin==1
    sortby='BIC';
end
sortbyname={'NLogL','BIC','AIC','AICc'};
if ~any(ismember(lower(sortby),lower(sortbyname)))
    oldvar=sortby; %May be 'PDF' or 'CDF' or other commands
    if isempty(varargin)
        varargin={oldvar};
    else
        varargin=[oldvar varargin];
    end
    sortby='BIC';
end
if nargin < 2, sortby='BIC'; end
distname={'beta', 'birnbaumsaunders', 'exponential', ...
    'extreme value', 'gamma', 'generalized extreme value', ...
    'generalized pareto', 'inversegaussian', 'logistic', 'loglogistic', ...
    'lognormal', 'nakagami', 'normal', ...
    'rayleigh', 'rician', 'tlocationscale', 'weibull'};
if ~any(strcmpi(sortby,sortbyname))
    error('allfitdist:SortBy','Sorting must be either NLogL, BIC, AIC, or AICc');
end
%Input may be mixed of numeric and strings, find only strings
vin=varargin;
strs=find(cellfun(@(vs)ischar(vs),vin));
vin(strs)=lower(vin(strs));
%Next check to see if 'PDF' or 'CDF' is listed
numplots=sum(ismember(vin(strs),{'pdf' 'cdf'}));
if numplots>=2
    error('ALLFITDIST:PlotType','Either PDF or CDF must be given');
end
if numplots==1
    plotind=true; %plot indicator
    indxpdf=ismember(vin(strs),'pdf');
    plotpdf=any(indxpdf);
    indxcdf=ismember(vin(strs),'cdf');
    vin(strs(indxpdf|indxcdf))=[]; %Delete 'PDF' and 'CDF' in vin
else
    plotind=false;
end
%Check to see if discrete
strs=find(cellfun(@(vs)ischar(vs),vin));
indxdis=ismember(vin(strs),'discrete');
discind=false;
if any(indxdis)
    discind=true;
    distname={'binomial', 'negative binomial', 'poisson'};
    vin(strs(indxdis))=[]; %Delete 'DISCRETE' in vin
end
strs=find(cellfun(@(vs)ischar(vs),vin));
n=numel(data); %Number of data points
data = data(:);
D=[];
%Check for NaN's to delete
deldatanan=isnan(data);
%Check to see if frequency is given
indxf=ismember(vin(strs),'frequency');
if any(indxf)
    freq=vin{1+strs((indxf))}; freq=freq(:);
    if numel(freq)~=numel(data)
        error('ALLFITDIST:PlotType','Matrix dimensions must agree');
    end
    delfnan=isnan(freq);
    data(deldatanan|delfnan)=[]; freq(deldatanan|delfnan)=[];
    %Save back into vin
    vin{1+strs((indxf))}=freq;
else
    data(deldatanan)=[];
end





%% Run through all distributions in FITDIST function
warning('off','all'); %Turn off all future warnings
for indx=1:length(distname)
    try
        dname=distname{indx};
        switch dname
            case 'binomial'
                PD=fitbinocase(data,vin,strs); %Special case
            case 'generalized pareto'
                PD=fitgpcase(data,vin,strs); %Special case
            otherwise
                %Built-in distribution using FITDIST
                PD = fitdist(data,dname,vin{:});
        end
        
        NLL=PD.NLogL; % -Log(L)
        %If NLL is non-finite number, produce error to ignore distribution
        if ~isfinite(NLL)
            error('non-finite NLL');
        end
        num=length(D)+1;
        PDs(num) = {PD}; %#ok<*AGROW>
        k=numel(PD.Params); %Number of parameters
        D(num).DistName=PD.DistName;
        D(num).NLogL=NLL;
        D(num).BIC=-2*(-NLL)+k*log(n);
        D(num).AIC=-2*(-NLL)+2*k;
        D(num).AICc=(D(num).AIC)+((2*k*(k+1))/(n-k-1));
        D(num).ParamNames=PD.ParamNames;
        D(num).ParamDescription=PD.ParamDescription;
        D(num).Params=PD.Params;
        D(num).Paramci=PD.paramci;
        D(num).ParamCov=PD.ParamCov;
        D(num).Support=PD.Support;
    catch err %#ok<NASGU>
        %Ignore distribution
    end
end
warning('on','all'); %Turn back on warnings
if numel(D)==0
    error('ALLFITDIST:NoDist','No distributions were found');
end





%% Sort distributions
indx1=1:length(D); %Identity Map
sortbyindx=find(strcmpi(sortby,sortbyname));
switch sortbyindx
    case 1
        [~,indx1]=sort([D.NLogL]);
    case 2
        [~,indx1]=sort([D.BIC]);
    case 3
        [~,indx1]=sort([D.AIC]);
    case 4
        [~,indx1]=sort([D.AICc]);
end
%Sort
D=D(indx1); PD = PDs(indx1);





%% Plot if requested
if plotind;
    plotfigs(data,D,PD,vin,strs,plotpdf,discind)
end


end





function PD=fitbinocase(data,vin,strs)
%% Special Case for Binomial
% 'n' is estimated if not given
vinbino=vin;
%Check to see if 'n' is given
indxn=any(ismember(vin(strs),'n'));
%Check to see if 'frequency' is given
indxfreq=ismember(vin(strs),'frequency');
if ~indxn
    %Use Method of Moment estimator
    %E[x]=np, V[x]=np(1-p) -> nhat=E/(1-(V/E));
    if isempty(indxfreq)||~any(indxfreq)
        %Raw data
        mnx=mean(data);
        nhat=round(mnx/(1-(var(data)/mnx)));
    else
        %Frequency data
        freq=vin{1+strs(indxfreq)};
        m1=dot(data,freq)/sum(freq);
        m2=dot(data.^2,freq)/sum(freq);
        mnx=m1; vx=m2-(m1^2);
        nhat=round(mnx/(1-(vx/mnx)));
    end
    %If nhat is negative, use maximum value of data
    if nhat<=0, nhat=max(data(:)); end
    vinbino{end+1}='n'; vinbino{end+1}=nhat;
end
PD = fitdist(data,'binomial',vinbino{:});
end





function PD=fitgpcase(data,vin,strs)
%% Special Case for Generalized Pareto
% 'theta' is estimated if not given
vingp=vin;
%Check to see if 'theta' is given
indxtheta=any(ismember(vin(strs),'theta'));
if ~indxtheta
    %Use minimum value for theta, minus small part
    thetahat=min(data(:))-10*eps;
    vingp{end+1}='theta'; vingp{end+1}=thetahat;
end
PD = fitdist(data,'generalized pareto',vingp{:});
end





function plotfigs(data,D,PD,vin,strs,plotpdf,discind)
%Plot functionality for continuous case due to Jonathan Sullivan
%Modified by author for discrete case

%Maximum number of distributions to include
%max_num_dist=Inf;  %All valid distributions
max_num_dist=4;

%Check to see if frequency is given
indxf=ismember(vin(strs),'frequency');
if any(indxf)
    freq=vin{1+strs((indxf))};
end

figure

%% Probability Density / Mass Plot
if plotpdf
    if ~discind
        %Continuous Data
        nbins = max(min(length(data)./10,100),50);
        xi = linspace(min(data),max(data),nbins);
        dx = mean(diff(xi));
        xi2 = linspace(min(data),max(data),nbins*10)';
        fi = histc(data,xi-dx);
        fi = fi./sum(fi)./dx;
        inds = 1:min([max_num_dist,numel(PD)]);
        ys = cellfun(@(PD) pdf(PD,xi2),PD(inds),'UniformOutput',0);
        ys = cat(2,ys{:});
        bar(xi,fi,'FaceColor',[160 188 254]/255,'EdgeColor','k');
        hold on;
        plot(xi2,ys,'LineWidth',1.5)
        legend(['empirical',{D(inds).DistName}],'Location','NE')
        xlabel('Value');
        ylabel('Probability Density');
        title('Probability Density Function');
        grid on
    else
        %Discrete Data
        xi2=min(data):max(data);
        %xi2=unique(x)'; %If only want observed x-values to be shown
        indxf=ismember(vin(strs),'frequency');
        if any(indxf)
            fi=zeros(size(xi2));
            fi((ismember(xi2,data)))=freq; fi=fi'./sum(fi);
        else
            fi=histc(data,xi2); fi=fi./sum(fi);
        end
        inds = 1:min([max_num_dist,numel(PD)]);
        ys = cellfun(@(PD) pdf(PD,xi2),PD(inds),'UniformOutput',0);
        ys=cat(1,ys{:})';
        bar(xi2,[fi ys]);
        legend(['empirical',{D(inds).DistName}],'Location','NE')
        xlabel('Value');
        ylabel('Probability Mass');
        title('Probability Mass Function');
        grid on
    end
else
     
%Cumulative Distribution
    if ~discind
        %Continuous Data
        [fi xi] = ecdf(data);
        inds = 1:min([max_num_dist,numel(PD)]);
        ys = cellfun(@(PD) cdf(PD,xi),PD(inds),'UniformOutput',0);
        ys = cat(2,ys{:});
        if max(xi)/min(xi) > 1e4; lgx = true; else lgx = false; end
        subplot(2,1,1)
        if lgx
            semilogx(xi,fi,'k',xi,ys)
        else
            plot(xi,fi,'k',xi,ys)
        end
        legend(['empirical',{D(inds).DistName}],'Location','NE')
        xlabel('Value');
        ylabel('Cumulative Probability');
        title('Cumulative Distribution Function');
        grid on
        subplot(2,1,2)
        y = 1.1*bsxfun(@minus,ys,fi);
        if lgx
            semilogx(xi,bsxfun(@minus,ys,fi))
        else
            plot(xi,bsxfun(@minus,ys,fi))
        end
        ybnds = max(abs(y(:)));
        ax = axis;
        axis([ax(1:2) -ybnds ybnds]);
        legend({D(inds).DistName},'Location','NE')
        xlabel('Value');
        ylabel('Error');
        title('CDF Error');
        grid on
    else
        %Discrete Data
        indxf=ismember(vin(strs),'frequency');
        if any(indxf)
            [fi xi] = ecdf(data,'frequency',freq);
        else
            [fi xi] = ecdf(data);
        end
        %Check unique xi, combine fi
        [xi,ign,indx]=unique(xi); %#ok<ASGLU>
        fi=accumarray(indx,fi);
        inds = 1:min([max_num_dist,numel(PD)]);
        ys = cellfun(@(PD) cdf(PD,xi),PD(inds),'UniformOutput',0);
        ys=cat(2,ys{:});
        subplot(2,1,1)
        stairs(xi,[fi ys]);
        legend(['empirical',{D(inds).DistName}],'Location','NE')
        xlabel('Value');
        ylabel('Cumulative Probability');
        title('Cumulative Distribution Function');
        grid on
        subplot(2,1,2)
        y = 1.1*bsxfun(@minus,ys,fi);
        stairs(xi,bsxfun(@minus,ys,fi))
        ybnds = max(abs(y(:)));
        ax = axis;
        axis([ax(1:2) -ybnds ybnds]);
        legend({D(inds).DistName},'Location','NE')
        xlabel('Value');
        ylabel('Error');
        title('CDF Error');
        grid on
    end
end

end
