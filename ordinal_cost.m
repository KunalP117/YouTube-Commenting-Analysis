function [x] = ordinal_cost(xs, prob_act, X, J, K)
%ORDINAL_COST Summary of this function goes here
%   Detailed explanation goes here
%Compute G_{w,k}
for w=1:K
    for k=1:K
        G(w+(k-1)*K) = prob_act((1+(w-1)*J):(J+(w-1)*J))*...
            xs((1+(k-1)*J+X*J*K):(J+(k-1)*J+X*J*K));
    end
end

%Constraint
A = zeros(K*(K-1),K);
rowcnt = 1;
for w=1:K
    for k=1:K
        if(w ~= k)
            A(rowcnt,k) = 1;
            A(rowcnt,w) = -1;
            b(rowcnt) = G(k+(k-1)*K)-G(w+(k-1)*K);
            rowcnt = rowcnt+1;
        end
    end
end

%Solve Linear Program
lb = ones(K,1).*1e-3;
ub = 10*ones(K,1);
f = zeros(K,1); % no objective function
x = linprog(f,A,b,[],[],lb,ub);


end

