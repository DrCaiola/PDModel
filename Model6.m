function [t,y,period,SS,rcond_error]=Model6(y0,w,T,Ba,M,varargin)
tic
warning('off','MATLAB:nearlySingularMatrix')
N=10^6;
opt_graph=1;
opt_rplot = 0;
opt_debug = 0;
SS_check=0;
ss_flag=0;
period=0;
TOL=10^-10;
stp=.01;
h=0;
flag=0;
E=zeros(6,6^3);

if nargin==0
   y0=zeros(6,1);
   w=zeros(6);
   w(1,2) = randi(30);
   w(6,2) = randi(30);
   w(1,6) = randi(30);
   w(3,2) = randi(30);
   w(3,6) = randi(30);
   w(4,2) = randi(30);
   w(4,6) = randi(30);
   w(5,2) = randi(30);
   w(4,3) = randi(30);
   w(3,4) = randi(30);
   w(4,5) = randi(30);
   %w1 = randi(30);
   %w2 = randi(30);
   %w(2,3) = w1-w2;
   w(2,3) = randi(30);
   T = 6*rand(1,6);
   Ba = [60+h 10 2 2 10 10] + 6*(rand(1,6)-.5);
   M = [300 100 50 50 100] + 5*(rand(1,5)-.5);
   %q = conditions(w,w(2,3),T);
end

%Ba(2) = Ba(2)-w(6,2)*Ba(6); Ba(6)=[]; %interneuron relay
for ii=1:6
    B(ii,:)=[0 Ba(ii) M(ii)];
end
numvarargs= length(varargin);
if numvarargs~=0
    for kk=1:numvarargs
        if strcmp(varargin{kk},'graphless')
            opt_graph=0;
        elseif strcmp(varargin{kk},'ss_check')
            SS_check = 1;
        elseif strcmp(varargin{kk},'debug')
            opt_debug = 1;
            opt_rplot = 1;
        else
            error(['"',num2str(varargin{kk}),'"', ' ', 'is not a valid option!'])
        end
    end
end



D=cell(3,3,3,3,3,3);
V=cell(3,3,3,3,3,3);
Beta=cell(3,3,3,3,3,3);
ss=cell(3,3,3,3,3,3);
RC=[];

f(1:3,1:6)  =...
    [-1 0 0 0 0 0;...
    -1 0 0 0 0 0;...
    -1 0 0 0 0 0];
f(1:3,7:12) =...
    [0 -1 0 0 0 0;...
    -w(1,2) -1 w(3,2) w(4,2) -w(5,2) -w(6,2);...
    0 -1 0 0 0 0];
f(1:3,13:18)=...
    [0 0 -1 0 0 0;...
    0 w(2,3) -1 w(4,3) 0 0;...
    0 0 -1 0 0 0];
f(1:3,19:24)=...
    [0 0 0 -1 0 0;...
    0 0 w(3,4) -1 0 0;...
    0 0 0 -1 0 0];
f(1:3,25:30)=...
    [0 0 0 0 -1 0;...
    0 0 0 w(4,5) -1 0;...
    0 0 0 0 -1 0];
f(1:3,31:36)=...
    [0 0 0 0 0 -1;...
    -w(1,6) 0 0 w(4,6) -w(5,6) -1;...
    0 0 0 0 0 -1];

kk=1;
for i1=1:3
    for i2=1:3
        for i3=1:3
            for i4=1:3
                for i5=1:3
                    for i6=1:3
                        A=[1/T(1)*f(i1,1:6);1/T(2)*f(i2,7:12);1/T(3)*f(i3,13:18);1/T(4)*f(i4,19:24);1/T(5)*f(i5,25:30);1/T(6)*f(i6,31:36)];
                        [V{i1,i2,i3,i4,i5,i6},D{i1,i2,i3,i4,i5,i6}]=eig(A);
                        E(1:6,kk) = eig(A);
                        RC(kk) = rcond(A);
                        kk=kk+1;
                        Beta{i1,i2,i3,i4,i5,i6}=[B(1,i1);B(2,i2);B(3,i3);B(4,i4);B(5,i5);B(6,i6)];
                        ss{i1,i2,i3,i4,i5,i6}=A\-diag([1/T(1),1/T(2),1/T(3),1/T(4),1/T(5),1/T(6)])*Beta{i1,i2,i3,i4,i5,i6};
                    end
                end
            end
        end
    end
end
%D{2,2,2,2,2}
if SS_check
    SS = ss{2,2,2,2,2,2};
    if length(find(abs(real(E))<10^(-12)))>=1
        flag=1;
    end
    t=0;
    y=0;
    return
end
y(:,1)=y0;
t(1)=0;
J=zeros(6,1);
I0=([f(2,1:6);f(2,7:12);f(2,13:18);f(2,19:24);f(2,25:30);f(2,31:36)]+eye(6))*y(:,1)+Ba';
for i0=1:6
    if I0(i0)>=M(i0)
        J(i0)=3;
    elseif I0(i0)<=0
        J(i0)=1;
    else
        J(i0)=2;
    end
end
if opt_debug
    disp("Intial Region " + num2str(J'));
end
k=1;
limitcheck=[y0' 0];
while k<N
    rcond_error = false;
    [t1,y1]=compute(y(:,k));
    
    y=[y y1(:,2:end)];

    t=[t t1(2:end)+t(end)];
    k=numel(t);
    if rcond_error
        k=N;
    end
    limitcheck=[limitcheck; y(:,end)' t(end)];
    for lct=1:size(limitcheck,1)-1
        if abs(limitcheck(lct,1:end-1)-limitcheck(end,1:end-1))<TOL
            period=limitcheck(end,end)-limitcheck(lct,end);
            %fprintf('Loop Detected at N=%d\n',k)
            k=N;
        end
    end
    if ss_flag
        k=N;
    end
    SS = ss;%{2,2,2,2,2,2};
end
if opt_graph
    figure;
    nucleiname={'GPe','TC','CT5','CT6','RTN','IN'};
    for ii=1:6
        subplot(6,1,ii);
        plot(t,1000.*y(ii,:))
        if opt_rplot
            ax = gca;
            hold on
            if size(limitcheck,1) >2
                plot(repmat(limitcheck(2:end-1,end),[1,2]),[0 ax.YLim(2)],'k')
            end
        end
        title([nucleiname{ii},' Firing Rate'])
        xlabel('time (ms)')
        ylabel('Frequency (Hz)')
        axis([0 t(end) 0 ceil(max(1000.*y(ii,:)))+.1*ceil(max(1000.*y(ii,:)))])
    end
end
    function [t,y]=compute(y0)
        Plane{1}={@(t,alpha) Ba(1)-alpha*M(1)};
        Plane{2}={@(t,alpha) -w(1,2)*planecor(1,t)+w(3,2)*planecor(3,t)...
            + w(4,2)*planecor(4,t)-w(5,2)*planecor(5,t)-w(6,2)*planecor(6,t)+Ba(2)-alpha*M(2)};
        Plane{3}={@(t,alpha) w(2,3)*planecor(2,t)+w(4,3)*planecor(4,t)+Ba(3)-alpha*M(3)};
        Plane{4}={@(t,alpha) w(3,4)*planecor(3,t)+Ba(4)-alpha*M(4)};
        Plane{5}={@(t,alpha) w(4,5)*planecor(4,t)+Ba(5)-alpha*M(5)};
        Plane{6}={@(t,alpha) -w(1,6)*planecor(1,t)+w(4,6)*planecor(4,t)...
            - w(5,6)*planecor(5,t)+Ba(6)-alpha*M(6)};

        
        C=@(t) diag([exp(D{J(1),J(2),J(3),J(4),J(5),J(6)}(1,1)*t),...
            exp(D{J(1),J(2),J(3),J(4),J(5),J(6)}(2,2)*t),exp(D{J(1),J(2),J(3),J(4),J(5),J(6)}(3,3)*t),...
            exp(D{J(1),J(2),J(3),J(4),J(5),J(6)}(4,4)*t),exp(D{J(1),J(2),J(3),J(4),J(5),J(6)}(5,5)*t),...
            exp(D{J(1),J(2),J(3),J(4),J(5),J(6)}(6,6)*t)]);
        chk = (V{J(1),J(2),J(3),J(4),J(5),J(6)}*C(0));
        c=chk\(y0-ss{J(1),J(2),J(3),J(4),J(5),J(6)});
        t(1)=0;
        t(2)=t(1)+stp;
        sum0=sum(J);
        y(:,1)=y0;
        if rcond(chk) <10^-10
            rcond_error = true;
            return
        end
        j=2;
        intJ=J;
        stk=0;
        while j<=N-k+1
            y(:,j)=real(V{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}*C(t(j))*c+ss{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)});
            I=([f(2,1:6);f(2,7:12);f(2,13:18);f(2,19:24);f(2,25:30);f(2,31:36);]+eye(6))*y(:,j)+Ba';
            for l=1:6
                if I(l)>=M(l)
                    J(l)=3;
                elseif I(l)<=0
                    J(l)=1;
                else
                    J(l)=2;
                end
            end
            if j>5
                ss_check=y(:,j)-y(:,j-1)+y(:,j)-y(:,j-2)+y(:,j)-y(:,j-3);
                if abs(ss_check)<TOL
                    ss_flag=1;
                    break
                end
            end
            i=find(J~=intJ);
            if ~isempty(i)
                if length(i)>1 || (abs(J(i)-intJ(i))>1)
                    if stk > 3
                        rcond_error = true;
                        break
                    else
                        t(j)=t(j)-.9*(t(j)-t(j-1));
                        stk=stk+1;
                        continue
                    end
                end
                if opt_debug
                    disp("Region " + num2str(J'));
                end
                if (intJ(i)==3) || (J(i)==3)
                    alpha=1;
                else
                    alpha=0;
                end
                g=cell2mat(Plane{i});
                g1= @(s) g(s,alpha);
                tstart=fzero(g1,[t(j-1) t(j)]);
                y0=real(V{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}*C(tstart)*c+ss{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)});
                y(:,j)=y0;
                t(j)=tstart;
                break
            else
                if j+1<=N-k+1
                    t(j+1)=t(j)+stp;
                end
                j=j+1;
            end
        end
        function X=planecor(i,t)
            X=real(...
                c(1)*V{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(i,1)*exp(D{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(1,1)*t)+...
                c(2)*V{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(i,2)*exp(D{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(2,2)*t)+...
                c(3)*V{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(i,3)*exp(D{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(3,3)*t)+...
                c(4)*V{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(i,4)*exp(D{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(4,4)*t)+...
                c(5)*V{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(i,5)*exp(D{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(5,5)*t)+...
                c(6)*V{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(i,6)*exp(D{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(6,6)*t)+...
                ss{intJ(1),intJ(2),intJ(3),intJ(4),intJ(5),intJ(6)}(i));
        end
    end
end