function [VS,VARS] = penalized_obj_var(x,lambda,u,pihat,mur,n,N, X, J,K)
%PENALIZED_OBJECTIVE Summary of this function goes here
%   Detailed explanation goes here
%Net utility value
for k=1:K
    V(k) = sum(u((1+(k-1)*X*J):(k*X*J)).*x((1+(k-1)*X*J):(k*X*J)).*mur((1+(k-1)*X*J):(k*X*J)));
end

%Penalized variance term
for k=1:K
    vec_val_nupihat = n((1+(k-1)*X*J):(k*X*J))...
        .*u((1+(k-1)*X*J):(k*X*J)).*(x((1+(k-1)*X*J):(k*X*J))./pihat((1+(k-1)*X*J):(k*X*J)));
    VM(k) = sum(vec_val_nupihat)/N(k);
    VAR(k) = sum(vec_val_nupihat-VM(k))/(N(k)-1);
end

VS = sum(V);
VARS = sum(VAR);

end

