function [r1,x] = milp_generalcost(J, K, X, cond_prob, prob_act)
%MILP_GENERALCOST Summary of this function goes here
%Rational inattention test

M = 1000;
%A and b are reserved for Ax <= b
A = sparse(J*(4*K*K-J-K)+1, X*J*K+2*J(K-1)+2*(J^2)*(K-1));
rowcnt = 1;
%Aeq and beq reserved for Aeqx = beq
Aeq = sparse(2*J*(K-1), X*J*K+2*J(K-1)+2*(J^2)*(K-1));
rowcnteq = 1;

%Binary indicies
intcon = [(1+2*J*(K-1)+X*J*K):(2*(J^2)*(K-1)+2*J*(K-1)+X*J*K)];
    
%Constraint 1
for j = 1:J
    for l=1:J
        for k=1:K
            if(j ~= l)
                s_jindx = 1+(j-1)*X+(k-1)*X*J;
                e_jindx = X+(j-1)*X+(k-1)*X*J;
                s_lindx = 1+(l-1)*X+(k-1)*X*J;
                e_lindx = X+(l-1)*X+(k-1)*X*J;
                A(rowcnt, s_jindx:e_jindx) = -cond_prob(s_jindx:e_jindx);
                A(rowcnt,  s_lindx:e_lindx) = cond_prob(s_jindx:e_jindx);
                b(rowcnt) = -1e-3; %small negative to force strict inequality for atleast one
                rowcnt = rowcnt+1;
            end
        end
    end
end

%Constraint 2
for j=1:J
    for l=1:J
        for k=1:(K-1)
            A(rowcnt, (j+(k-1)*J+X*J*K)) = -1;
            A(rowcnt, (1+(l-1)*X+(k-1)*X*J):(X+(l-1)*X+(k-1)*X*J)) = cond_prob((1+(j-1)*X+(k-1)*X*J):(X+(j-1)*X+(k-1)*X*J));
            b(rowcnt) = 0;
            rowcnt = rowcnt+1;
        end
    end
end
            
%Constraint 3
for j=1:J
    for l=1:J
        for k=1:(K-1)
            A(rowcnt, (1+(l-1)*X+(k-1)*X*J):(X+(l-1)*X+(k-1)*X*J)) = -cond_prob((1+(j-1)*X+(k-1)*X*J):(X+(j-1)*X+(k-1)*X*J));
            A(rowcnt, l+(j-1)*J+(k-1)*(J^2)+2*J*(K-1)+X*J*K) = M;
            A(rowcnt, j+(k-1)*J+X*J*K) = 1;
            b(rowcnt) = M;
            rowcnt = rowcnt+1;
        end
    end
end

%Constraint 4 (equality)
for j=1:J
    for k=1:(K-1)
        Aeq(rowcnteq, (1+(j-1)*J+(k-1)*(J^2)+2*J(K-1)+X*J*K):(J+(j-1)*J+(k-1)*(J^2)+2*J(K-1)+X*J*K)) = 1;
        beq(rowcnteq) = 1;
        rowcnteq = rowcnteq + 1;
    end
end

%Constraint 5
for j=1:J
    for l=1:J
        for k=1:(K-1)
            if(k <= (K-2))
                A(rowcnt, (j+k*J+X*J*K+J*(K-1))) = -1;
                A(rowcnt, (1+(l-1)*X+(k-1)*X*J):(X+(l-1)*X+(k-1)*X*J)) = cond_prob((1+(j-1)*X+k*X*J):(X+(j-1)*X+k*X*J));
            else
                A(rowcnt, (j+X*J*K+J*(K-1))) = -1;
                A(rowcnt, (1+(l-1)*X+(k-1)*X*J):(X+(l-1)*X+(k-1)*X*J)) = cond_prob((1+(j-1)*X):(X+(j-1)*X));
            end
            b(rowcnt) = 0;
            rowcnt = rowcnt+1;
        end
    end
end
                
%Constraint 6
for j=1:J
    for l=1:J
        for k=1:(K-1)
            if(k <= (K-2))
                sl_indx = 1+(l-1)*X+(k-1)*X*J;
                el_indx = X+(l-1)*X+(k-1)*X*J;
                sj_indx = 1+(j-1)*X+k*X*J;
                ej_indx = X+(j-1)*X+k*X*J;
                zeta_indx = l+(j-1)*J+k*(J^2)+(K-1)*(J^2)+2*J*(K-1)+X*J*K;
                A(rowcnt, j+k*J+X*J*K+J*(K-1)) = 1;
                A(rowcnt, sl_indx:el_indx) = -cond_prob(sj_indx:ej_indx);
                A(rowcnt, zeta_indx) = M;
            else
                sl_indx = 1+(l-1)*X+(k-1)*X*J;
                el_indx = X+(l-1)*X+(k-1)*X*J;
                sj_indx = 1+(j-1)*X;
                ej_indx = X+(j-1)*X;
                zeta_indx = l+(j-1)*J+(K-1)*(J^2)+2*J*(K-1)+X*J*K;
                A(rowcnt, j+X*J*K+J*(K-1)) = 1;
                A(rowcnt, sl_indx:el_indx) = -cond_prob(sj_indx:ej_indx);
                A(rowcnt, zeta_indx) = M;
            end
            b(rowcnt) = M;
            rowcnt = rowcnt+1;
        end
    end
end
                                                
%Constraint 7 (equality)
for j=1:J
    for k=1:(K-1)
        if(k <= (K-2))
            sl_indx = (1+(j-1)*J+k*(J^2)+(J^2)*(K-1)+2*J*(K-1)+X*J*K);
            el_indx = (J+(j-1)*J+k*(J^2)+(J^2)*(K-1)+2*J*(K-1)+X*J*K);
            Aeq(rowcnteq, sl_indx:el_indx) = 1;
        else
            sl_indx = (1+(j-1)*J+(J^2)*(K-1)+2*J*(K-1)+X*J*K);
            el_indx = (J+(j-1)*J+(J^2)*(K-1)+2*J*(K-1)+X*J*K);
            Aeq(rowcnteq, sl_indx:el_indx) = 1;
        end
        beq(rowcnteq) = 1;
        rowcnteq = rowcnteq+1;
    end
end
    
%Constraint 8
for k=1:(K-1)
    if(k <= (K-2))
        A(rowcnt, (1+(k-1)*J+X*J*K):(J+(k-1)*J+X*J*K)) = -prob_act((1+(k-1)*J):(J+(k-1)*J));
        A(rowcnt, (1+k*J+X*J*K+J*(K-1)):(J+k*J+X*J*K+J*(K-1))) = -prob_act((1+k*J):(J+k*J));
    else
        A(rowcnt, (1+(k-1)*J+X*J*K):(J+(k-1)*J+X*J*K)) = -prob_act((1+(k-1)*J):(J+(k-1)*J));
        A(rowcnt, (1+X*J*K+J*(K-1)):(J+X*J*K+J*(K-1))) = -prob_act(1:J);
    end
    b(rowcnt) = 0;
end

r1=0;r2=0;r3=0;
%%
%Solve MILP to construct utility function
lb = zeros(X*J*K+2*J(K-1)+2*(J^2)*(K-1),1);
%lb(1:(X*J*K)) = 1e-3;
ub = ones(X*J*K+2*J(K-1)+2*(J^2)*(K-1),1);
%ub(1:(X*J*K)) = 100;
f = zeros(X*J*K+2*J(K-1)+2*(J^2)*(K-1),1); % no objective function
%options = optimoptions('intlinprog','Display','iter', 'IntegerTolerance', 1e-6);
[r1 r2 x r3] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub);
%y=[transpose(A*x)-b transpose(Aeq*x)-beq] ;

end

