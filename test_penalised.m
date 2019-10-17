clear all;
clc;


%Parameters of optimization problem
J = 6; % comments ({low count, negative; low count, neutral; low count, positive;
                %high count, negative; high count, neutral; high count, positive;
K = 2; % decision problems (category of video {most popular; other})
X = 2; % viewcount ({high, low})
cl_MAX=153;
i=0;

x=0;
e=0;
tot=0;gr_tot=0;
count=0;count_1=0;
arr_1=zeros(1,1);
arr_2=zeros(1,1);

 A=0;b=0;Aeq=0;beq=0;
 beta=0;
 good_beta_arr=zeros(2,2);
 ans=0;
 
 
 %Construct utility for each frame
 load('category_switch_data_mod_zero_rec_1.mat');
 %load('80_percent_data_prob.mat');
arr_3=zeros(10,10,X*J*K);
arr=zeros(10,X*J*K);
dummy=0;dumm_1=0;
% for beta = 0.01:0.01:0.95
%     count_1=count_1+1;
%     count=0;
%     count_2=0;
%     tot=0;
%     %count
%     for i=1:17
%         for j=(i+1):18    
%             count=count+1;
%             [x,e,ans] = milp_reynicost(J, K, X, cond_prob(count,:), prob_act(count,:),joint_prob(count,:),state_prob(count,:),beta);
%             if e==1
%                 tot=tot+1;
%                 [x,e] = milp_reynicost_kkt_dev(J, K, X, cond_prob(count,:), prob_act(count,:),joint_prob(count,:),state_prob(count,:),beta);
%                 if e==1
%                     count_2=count_2+1;
%                    arr(count_2,:)=x(X*J*K+2*J(K-1)+2*(J^2)*(K-1)+2+2 + 1 : X*J*K+2*J(K-1)+2*(J^2)*(K-1)+2+2 + X*J*K)/norm(x(1:X*J*K));
%               end
%             end
%         end
%     end
% 
%     %arr_3(count_1,:,:)= arr;
%     tot    
%    good_beta_arr(count_1,:)=[beta tot/153];
% end


arr_3=zeros(1,153);
count=0;
count_1=0;
count_2=0;
tot=0;
fval=0;
global condprob; global probact; global jointprob; global stateprob;
x_arr=zeros(45,X*J*K + J*K + 4);
fval_arr=zeros(45,1);
grad_shannon = zeros(45,24);
%for beta=0.01:0.01:0.99
count_2=count_2+1;
count=0;

count_1=0;
count_2=0;
val_arr_1=zeros(1,100);
val_arr_2=zeros(1,100);
for i=1:17
    for j=(i+1):18   
        count=count+1;
        condprob=cond_prob(count,:);
        jointprob=joint_prob(count,:);
        probact=prob_act(count,:);
        stateprob=state_prob(count,:);
        if (i==1 || i==5 || i==7 || i==8 || i==9 || i==10 || i==11 || i==12 || i==14 || i==17) && ( j==1 || j==5 || j==7 || j==8 || j==9 || j==10 || j==11 || j==12 || j==14 || j==17) && (i~=j)
            count_1=count_1+1;
            [x,e,val_arr_1(count_1)] = milp_reynicost_niacs_dev_true(J, K, X, condprob, probact); 
            %if e~=0
            %    val_arr_1(count_1)=val_arr_1(count_1)/(0.5*(sum(x(1:12).^2) + sum(x(13:24).^2)));
            %end
        elseif (i==2 || i==3 || i==4 || i==6 || i==13 || i==15 || i==16 || i==18) && ( j==2 || j==3 || j==4 || j==6 || j==13 || j==15 || j==16 || j==18) && (i~=j)
         
        %else
            count_2=count_2+1;
            [x,e,val_arr_2(count_2)] = milp_reynicost_niacs_dev_post(J, K, X, condprob, probact); 
            if e~=0
                val_arr_2(count_2)=val_arr_2(count_2)/(0.5*(sum(x(1:12).^2) + sum(x(13:24).^2)));
            end
        else
        end
    end
end


% arr_3=zeros(1,153);
% count=0;
% count_1=0;
% tot=0;
% fval=0;
% global condprob; global probact; global jointprob; global stateprob;
% x_arr=zeros(10,X*J*K + J*K + 4);
% fval_arr=zeros(10,1);
% grad_shannon=zeros(28,24);
% for i=1:17
%     for j=(i+1):18   
%         count=count+1;
%         condprob=cond_prob(count,:);
%         jointprob=joint_prob(count,:);
%         probact=prob_act(count,:);
%         stateprob=state_prob(count,:);
%         if (i==1 || i==5 ||  i==7 || i==8 || i==9 || i==10 || i==11 || i==12 || i==14 || i==17) && ( j==1 || j==5 || j==7 || j==8 || j==9 || j==10 || j==11 || j==12 || j==14 || j==17)
%             [x,e] = milp_shannoncost_niacs(J, K, X, condprob, probact);
%         %if e==1   
%             x=x';
%             count_1=count_1+1;
%             [x,fval,e,aux_arr]= milp_shannoncost_kkt_dev(J, K, X,jointprob,condprob, probact,x);
%             x_arr(count_1,:) = x;
%             fval_arr(count_1)= fval;
%             %aux_arr
%             grad_shannon(count_1,:)=aux_arr;
%         end
%     end
% end

% beta=0.4;
% count=0;
% tot=0;
% dev_arr=zeros(1,153);
% for i=1:17
%     for j=(i+1):18   
%         count=count+1;
%         [x,e] = milp_reynicost_niacs_dev(J, K, X, cond_prob(count,:), prob_act(count,:), joint_prob(count,:), state_prob(count,:),beta);
%         if e==1         
%             dev_arr(count)=x(X*J*K+2*J(K-1)+2*(J^2)*(K-1)+2+2 + 1);           
%         end
%     end
% end

% count_1=0;
% beta_pass=zeros(1,99);
% for beta = 0.01:0.01:0.99
%     beta
%     count=0;
%     tot=0;
%     count_1=count_1+1;
%     for i=1:17
%         for j=(i+1):18   
%             count=count+1;
%             [x,e,ans] = milp_reynicost(J, K, X,cond_prob(count,:), prob_act(count,:), joint_prob(count,:), state_prob(count,:), beta);
%             if e==1 || e==2
%               tot=tot+1;
%               arr_1(count_1,tot)=i;
%               arr_2(count_1,tot)=j;
%             end
%         end
%     end
%     beta_pass(count_1)=tot; 
% end

%General Cost - 45% of combinations, 10/18 categories
%Shannon Cost - <1% of combinations, 02/18 categories

% load('frame2_prob_ext.mat')
% x2 = milp_generalcost(J, K, X, cond_prob, prob_act);
% c2 = ordinal_cost(x2, prob_act, X, J, K);
% load('frame3_prob_ext.mat')
% x3 = milp_generalcost(J, K, X, cond_prob, prob_act);
% c3 = ordinal_cost(x3, prob_act, X, J, K);
% load('frame4_prob_ext.mat')
% x4 = milp_generalcost(J, K, X, cond_prob, prob_act);
% c4 = ordinal_cost(x4, prob_act, X, J, K);