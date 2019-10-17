function [x,exitflag,ans] = milp_reynicost(J, K, X, cond_prob, prob_act, joint_prob, state_prob, beta)
%MILP_GENERALCOST Summary of this function goes here
% %Normalize the joint_prob
% for k=1:K
%     joint_prob((1:X*J)+(k-1)*X*J) = joint_prob((1:X*J)+(k-1)*X*J)/sum(joint_prob((1:X*J)+(k-1)*X*J));
% end
% for i=1:X
%     for k=1:K
%         state_prob(i+(k-1)*K) = sum(joint_prob((i+(k-1)*X*J):(i+J+(k-1)*X*J)));
%     end
% end
%Rational inattention test
M = 10000;
%A and b are reserved for Ax <= b
A = sparse(J*(4*K*K-J-K)+1, X*J*K+2*J(K-1)+2*(J^2)*(K-1)+2+2);
rowcnt = 1;
%Aeq and beq reserved for Aeqx = beq
Aeq = sparse(2*J*(K-1)+X*J*K, X*J*K+2*J(K-1)+2*(J^2)*(K-1)+2+2);
rowcnteq = 1;

%Binary indicies
intcon = [(1+2*J*(K-1)+X*J*K):(2*(J^2)*(K-1)+2*J*(K-1)+X*J*K)];
    
%Constraint 1 NIAS
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

%Constraint 2 m_k >=
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
            
%Constraint 3 M (1) (mk)
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

%Constraint 4 (equality) sum = 1
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
                
%Constraint 6  M (2) n_k
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
                                                
%Constraint 7 (equality) sum = 1
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
    
%Constraint 8 NIAC
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

dumm1=0;
dumm2=0;
%Constraint 9 (equality) KKT
for i=1:X
    for j=1:J
        for k=1:K
            coeff = 0;
            for si=1:X
                for sj=1:J
                    %coeff = coeff+joint_prob(si+(sj-1)*X+(k-1)*X*J)*((cond_prob(si+(sj-1)*X+(k-1)*X*J)/state_prob(si+(k-1)*K))^(beta-1));
                    coeff = coeff+(joint_prob(si+(sj-1)*X+(k-1)*X*J)^(beta))/[(state_prob(si+(k-1)*K)^(beta-1))*(prob_act((k-1)*J + sj)^(beta-1))];
                end
            end
            %rectify KKT for renyi case
            %coeff = coeff*((cond_prob(i+(j-1)*X+(k-1)*X*J)/state_prob(i+(k-1)*K)^(beta-1)))/(beta-1);
            %coeff=coeff*(beta-1)*(prob_act( j+(k-1)*J ))^(2*(beta-1));
            if joint_prob(1 +(j-1)*X+(k-1)*X*J) == 0
                joint_prob(1 +(j-1)*X+(k-1)*X*J) = 0.000001;
            end
            if joint_prob(2 +(j-1)*X+(k-1)*X*J) == 0
                joint_prob(2 +(j-1)*X+(k-1)*X*J) = 0.000001;
            end
            if prob_act(j+(k-1)*J) == 0
                prob_act(j+(k-1)*J) = 0.000001;
            end
            dumm1= [(beta)*(prob_act(j+(k-1)*J)^(beta-1))*(joint_prob(i+(j-1)*X+(k-1)*X*J)^(beta-1)) - (beta-1)*(prob_act(j+(k-1)*J)^(beta-2))*(joint_prob(i+(j-1)*X+(k-1)*X*J)^(beta))]/(state_prob(i+(k-1)*X)^(beta-1));
            dumm2= [(beta-1)*(prob_act(j+(k-1)*J)^(beta-2))* (joint_prob(mod(i,2)+1 +(j-1)*X+(k-1)*X*J)^(beta))]/(state_prob(mod(i,2)+1 +(k-1)*X)^(beta-1)); % can use the mod trick just because we have two states 1 and 2
            
            if ~isnan(coeff)
                coeff=(dumm1 - dumm2)/[coeff*(beta-1)*(prob_act( j+(k-1)*J ))^(2*(beta-1))];
            else
                if dumm1>=dumm2
                    coeff=Inf;
                else
                    coeff= -Inf;
                end
            end
            %coeff
            Aeq(rowcnteq, i+(j-1)*X+(k-1)*X*J) = 1;
            %if ~(isnan(coeff))
                Aeq(rowcnteq, (2*(k-1))+1+2*J*(K-1)+2*(J^2)*(K-1)+X*J*K) = -coeff;
            %else
            %   Aeq(rowcnteq, 1+2*J*(K-1)+2*(J^2)*(K-1)+X*J*K) = (-1)*M;
            %end
            %if cond_prob(i+(j-1)*X+(k-1)*X*J) ~= 0
            %    Aeq(rowcnteq, 1+2*J*(K-1)+2*(J^2)*(K-1)+X*J*K) = -log(cond_prob(i+(j-1)*X+(k-1)*X*J));
            %else 
            %    Aeq(rowcnteq, 1+2*J*(K-1)+2*(J^2)*(K-1)+X*J*K) = 0;
            %end
            Aeq(rowcnteq, (2*(k-1))+2+2*J*(K-1)+2*(J^2)*(K-1)+X*J*K) = 1;
            beq(rowcnteq) = 0;
            rowcnteq = rowcnteq+1;
        end
    end
end


%Solve MILP to construct utility function
lb = zeros(X*J*K+2*J(K-1)+2*(J^2)*(K-1)+2+2,1);
lb(X*J*K+2*J(K-1)+2*(J^2)*(K-1)+2) = -Inf;
lb(X*J*K+2*J(K-1)+2*(J^2)*(K-1)+2+2) = -Inf;
%lb(1:(X*J*K)) = 1e-3;
ub = ones(X*J*K+2*J(K-1)+2*(J^2)*(K-1)+2+2,1);
ub(X*J*K+2*J(K-1)+2*(J^2)*(K-1)+1:X*J*K+2*J(K-1)+2*(J^2)*(K-1)+2+2) = [Inf Inf Inf Inf];
%ub(1:(X*J*K)) = 100;
f = zeros(X*J*K+2*J(K-1)+2*(J^2)*(K-1)+2+2,1); % no objective function
%options = optimoptions('intlinprog','Display','iter', 'IntegerTolerance', 1e-6);
[x,~,exitflag] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub);
Aeq=full(Aeq);
ans=horzcat(transpose(Aeq(13:2:35,109)),transpose(Aeq(14:2:36,111)));
%Aeq
end
