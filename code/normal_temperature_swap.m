%---------------------------------------------------------------------
%
% Toy parallel tempering. multiple source inversion. 
% 
% Juan and Janos
%
%
%   -Delta u(x)=f(x,y;theta) in D,
%   u(x,y)=0                on dD 
%   f(x,y;theta)=exp(-(x-x0)^2-(y-y0)^2) + exp(-(x-x1)^2-(y-y1)^2)
%   where theta=(x0,y0), D=[0,1]^2
%
%
%---------------------------------------------------------------------
clc;
clear variables; 
close all;
%---------------------------------------------------------------------
%
% Generates numerical pde related stuff
%
%---------------------------------------------------------------------
n=64;       %number of grid points on each component     
h=1/n;
e = ones(n,1);
k = spdiags([-e 2*e -e], -1:1, n, n);
I=eye(n);
A=(kron(k,I)+kron(I,k))/(h^2);
x=linspace(0,1,n);
y=linspace(0,1,n)';
%defines function to call the right hand side
F=@(X) 0.5*reshape(exp(-1000*(x-X(1)).^2-1000*(y-X(2)).^2) + exp(-1000*(x-X(3)).^2-1000*(y-X(4)).^2),n^2,1);
%---------------------------------------------------------------------
%
% Generates Data
%
%---------------------------------------------------------------------
%---------------------------------------------------------------------
%
% Generates Data
%
%---------------------------------------------------------------------
th_true=[.7,.7,.3,.3];
f=F(th_true);
u_true=A\f;
[xg,yg]=meshgrid(x,y);
F=@(X) 0.5*reshape(exp(-1000*(x-X(1)).^2-1000*(y-X(2)).^2) + exp(-1000*(x-X(1)).^2-1000*(y-X(2)).^2),n^2,1);


%for simplicity, let's use the whole solution as an observation operator
sigma=0.01*max(u_true);
d=u_true+sigma*randn(length(u_true),1); %adds noise to the solution

%plots true data, just becuase
figure(1)
subplot(121)
surf(xg,yg,reshape(u_true,n,n));
title('u_t(x,\theta_t)')
subplot(122)
surf(xg,yg,reshape(d,n,n));
title('d(x,\theta_t)')



%---------------------------------------------------------------------
%
% Defines some hyper parameters, Number of temperatures, N samples, etc.  
%
%---------------------------------------------------------------------
p=2;%number of parameters
N=10^4; %number of samples
N_temp=2;%number of temperatures
Ns=1; %How often do we swap
X=cell(N_temp,1);%prealloactes
y=zeros(N_temp,p); %preallocates proposals
%beta=20.^(0:N_temp); %vector of inverse temperatures\
beta = [1 5];
acpt=zeros(N_temp,1); %acceptace rate
%lower and upper limit for uniform prior
a=-5;
b=5; 
%Defines likelihood, prior and posterior functions

%time=linspace(0,3,length(d));
%L=@(x)  -0.5*trapz(time,abs(d-f(x)).^2)/sigma^2;
L=@(x)  log(0.5*normpdf(x,-1,sigma)+0.5*normpdf(x,2,sigma));

rho = @(x1,x2,beta1,beta2) exp((L(x1)/beta1)+(L(x2)/beta2))/(exp((L(x1)/beta1)+(L(x2)/beta2))+exp((L(x2)/beta1)+(L(x1)/beta2)));


%pr=@(x) log(unifpdf(x,a,b));
%post=@(x) L(x)+pr(x);
ratio=zeros(N_temp,1);
%preallocates matrizx to store log-posteriors
px=zeros(N,N_temp);
py=zeros(N,N_temp);
%preallocates samples and gives them different initial values 
for i=1:N_temp
X{i}=zeros(N,p);
X{i}(1,:)=a+(b-a).*rand(1,p);
px(1,i)=L(X{i}(1,:));
%px(1,i)=post(X{i}(1,:));

end
weight{1}(1) = rho(X{1}(1,:),X{2}(1,:),1,5);
weight{2}(1) = rho(X{2}(1,:),X{1}(1,:),1,5);
sigma_rwm=[0.5,2];%*linspace(1,100,N_temp);
disp('Entered MH loop')
delta=zeros(N,2);
delta(1,:) = [1,2];
for j=1:N-1

    
    
    
    for k=1:N_temp
        y(k,:)=X{i}(j,:)+sigma_rwm(k)*randn(1,p);
        %py(j,k)=post(y(k,:));
        py(j,k)=L(y(k,:));

    end


%accepts-rejects

for i=1:N_temp
    ratio(i)=min(0,py(j,i)/beta(i)-px(j,i)/beta(i));
    if log(rand)<ratio(i) %&& isnan((py(j,i)/beta(i))-px(j,i)/beta(i))==0 && prior(y(i,:))~=0
        X{i}(j+1,:)=y(i,:);
        px(j+1,i)=py(j,i);
        acpt(i)=acpt(i)+1;
    else
         X{i}(j+1,:)=X{i}(j,:);
         px(j+1,i)=px(j,i);
    end
end

%disp([' ratio ',num2str(exp(ratio'))])

% %swaps
% if mod(j,Ns)==0
%    for k=1:N_temp
%        if k<N_temp
%        ak=px(j+1,k+1)/beta(k)+px(j+1,k)/beta(k+1)-px(j+1,k)/beta(k)-px(j+1,k+1)/beta(k+1);
%        if ak>log(rand)
%          %  disp(['swapped temps ' , num2str(k) ,' and ',num2str(k+1)])
%            %changes posteriors
%            p1=px(j+1,k);p2=px(j+1,k+1);
%            px(j+1,k)=p2; px(j+1,k+1)=p1;
%            %changes points
%            p1=X{k}(j+1,:);   p2=X{k+1}(j+1,:); 
%           X{k}(j+1,:)=p2;   X{k+1}(j+1,:)=p1; 
%        end
%  
%        end
%    end
% end
   
 

%swaps temperatures

 bp=beta(2);
 bb=beta(1);
 beta(1)=bp; beta(2)=bb;
 %delta(j+1,:)=fliplr(delta(j,:));
 sigma_rwm = fliplr(sigma_rwm);
 
 weight{1}(j+1) = rho(X{1}(j+1,:),X{2}(j+1,:),1,5);
 
 weight{2}(j+1) = rho(X{2}(j+1,:),X{1}(j+1,:),1,5);
end
% Ya=[];
% Yb=[];
% Ya(:,1)=X{1}(delta(:,1)==1,:);
% Ya(:,2)=X{2}(delta(:,2)==1,:);
% Ya=Ya(:);
% 
% Yb(:,1)=X{1}(delta(:,1)==2,:);
% Yb(:,2)=X{2}(delta(:,2)==2,:);
% Yb=Yb(:);
% 


Ya = [X{1}, X{2}];
Wa = [weight{1}, weight{2}];

Yb = [X{2}, X{1}];
Wb = [weight{1}, weight{2}];


for j = 1:10000
   i=randsample(2,1,true,[weight{1}(j),weight{2}(j)]);
   Ya_res(j) = X{i}(j,:);
   Yb_res(j) = X{3-i}(j,:);
end


figure(10)
for i=1:N_temp
    subplot(N_temp/2,2,i);
    plot(X{i});
end
figure(20)
for i=1:N_temp
    [ff,x]=ksdensity(X{i});
    plot(x,(ff));hold on;
end


figure(30)
[ff,x]=ksdensity(Ya_res);
plot(x,(ff));hold on;
[ff,x]=ksdensity(Yb_res);
plot(x,(ff));hold off;

