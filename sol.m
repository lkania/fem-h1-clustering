%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = X;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fix Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

itmax = 1e6;
tol = 1e-10;
trials = 5;
do_analysis = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L-curve per K and K analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_analysis
    
    logepssqrs = [-10:1:1];
    epssqrs = 10.^logepssqrs;
    
    ks = 2:6;
    Ls = zeros(length(epssqrs),length(ks));
    Llins = zeros(length(epssqrs),length(ks));
    Lquads = zeros(length(epssqrs),length(ks));
        
    for K=ks
        for i = 1:length(epssqrs)
            for t = 1:trials
                eps = epssqrs(i);
                [S, gamma, L, Llin, Lquad]=femh1(x,K,itmax,eps,tol,false);

                disp(['K = ' num2str(K) '. Trial = ' num2str(t) '. L = ' num2str(L)])

                if t == 1 || (t > 1 && L < Ls(i,K-1))
                    Ls(i,K-1) = L;
                    Llins(i,K-1) = Llin;
                    Lquads(i,K-1) = Lquad;
                end
            end

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % L-curve per K plot
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        f = figure;
        subplot(1,2,1)
        str = sprintf('L-curve for K=%i',K);
        title(str)
        plot(Lquads(:,K-1),Llins(:,K-1),'bo-','linewidth',1.5)
        ylabel('Modelling Error')
        xlabel('Smoothness')
        subplot(1,2,2)
        title('L-curve')
        plot(logepssqrs,Llins(:,K-1),'bo-','linewidth',1.5)
        ylabel('Modelling Error')
        xlabel('Regularization')
        str = sprintf('%i-lcurve.png',K);
        saveas(f,str);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Unified L-curve plot
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %colorstring = 'kbgrym';
    colormap default;
    cmap=colormap;
    
    f = figure;
    subplot(1,2,1)
    title('L-curves')
    ylabel('Modelling Error')
    xlabel('Smoothness')
    hold on

    for K=ks
        color = cmap(K*10,:);
        plot(Lquads(:,K-1),Llins(:,K-1),'bo-','linewidth',1.5,'DisplayName',string(K),'Color', color)
    end
    hold off
    legend()    

    subplot(1,2,2)
    title('L-curves')
    ylabel('Modelling Error')
    xlabel('Regularization')

    hold on
    for K=ks
        color = cmap(K*10,:);
        plot(logepssqrs,Llins(:,K-1),'bo-','linewidth',1.5,'DisplayName',string(K),'Color', color)
    end
    hold off
    legend()
    
    saveas(f,'lcurves.png');   
    
    f = figure ;
    subplot(1,2,1)
    title('Modelling Error per K')
    ylabel('Modelling Error')
    xlabel('K')
    hold on
    
    for i = 1:length(epssqrs)
        color = cmap(i*5,:);
        plot(ks,Llins(i,:),'bo-','linewidth',1.5,'DisplayName',string(epssqrs(i)),'Color', color)
    end
    hold off
    legend()
    
    subplot(1,2,2)
    title('Modelling Error per K')
    ylabel('Improvement(%)')
    xlabel('K')
    hold on
    
    for i = 1:length(epssqrs)
        color = cmap(i*5,:);
        improvement = Llins(i,:)-circshift(Llins(i,:),1);
        improvement(1)=0;
        improvement = improvement ./ circshift(Llins(i,:),1);
        improvement = abs(improvement);
        
        plot(ks,improvement,'bo-','linewidth',1.5,'DisplayName',string(epssqrs(i)),'Color', color)
    end
    hold off
    legend()

    saveas(f,'errorperK.png');   

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plots S and gamma for the best configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eps = 10^-2;
K = 3;
[S, gamma, L, Llin, Lquad]=femh1(x,K,itmax,eps,tol,true);

[n,T] = size(x);

f = figure;

for k=1:K
    subplot(K,1,k)
    plot(gamma((k-1)*T+1:k*T))
    str = sprintf('S(%d) %s',k,strjoin(cellstr(num2str(S(:,k))),', '));
    title(str);
    ylim([-0.2 1.2])
end

saveas(f,'bestparams.png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FEM-H1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [S, gamma, L, Llin, Lquad]=femh1(x,K,itmax,eps,tol,print_)
   

    [n,T] = size(x);

    % init gamma with valid probability distributions

    gamma = rand(K*T,1);
    gamma_s = zeros(T,1);
    for k=1:K
        gamma_s = gamma_s + gamma((k-1)*T+1:k*T);
    end
    for k=1:K
        gamma((k-1)*T+1:k*T) = gamma((k-1)*T+1:k*T) ./ gamma_s;
    end

    % create B 

    B = sparse(repmat(eye(T),1,K));

    % create c

    c = ones(T,1);

    % create H_hat

    a = ones(T,1);
    diags_ = horzcat(-a,2*a,-a);
    H = spdiags(diags_,[-1 0 1],T,T);
    H(1,1) = 1;
    H(end,end) = 1;
    H_hat = kron(eye(K),H);
    H_hat = eps * H_hat;

    % variables for iterations
    S = zeros(n,K); 
    l = zeros(K*T,1); % lower bound for quadprog
    g = zeros(K*T,1); 
    L = Inf;
    
    ops = optimoptions('quadprog','Display','none', ... 
        'Algorithm','interior-point-convex', ...
        'OptimalityTolerance',1e-12, ...
        'MaxIterations',1e4);

    it = 0;
    while it < itmax

        % S-problem

        for k=1:K
            val = zeros(n,1);
            denominator = sum(gamma((k-1)*T+1:k*T));
            if denominator ~= 0
                val = sum((repmat(gamma((k-1)*T+1:k*T).',n,1) .* x),2) ./ denominator;
            end
            S(:,k) = val;
        end

        % gamma-problem

        % compute g

        for t = 1:T
            for k = 1:K
                g((k-1)*T+t) = sum((x(:,t) - S(:,k)).^2); 
            end
        end
        g = g/(T*n);

        % solve quad. problem

        [gamma, qpit] = quadprog(H_hat,g,[],[],B,c,l,[],gamma,ops);

        Lold = L; 
        Llin = dot(g,gamma);
        Lquad = dot(H_hat*gamma,gamma);
        L = Llin + eps*Lquad; 
        deltaL = Lold - L;
        
        % display progress

        if print_
            disp([num2str(it) '. it: quadprog_it = ' num2str(qpit) ', deltaL = ' num2str(deltaL)])
        end

        it = it + 1;

        if abs(deltaL) < tol
            break; 
        end

    end

end

