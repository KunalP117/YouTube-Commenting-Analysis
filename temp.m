 clear all;
 clc;

%load('category_switch_mod_1.mat');
load('20_percent_data_prob.mat');
% Aeq=zeros(24,4);
% beq=zeros(1,4);
% f=zeros(1,4);
% 
% for i=1:12
%     A(i,1)=ans(i,1); A(i,2)=-1;
%     A(i+12,3)=ans(i+12);A(i+12,4)=-1;
%     beq(i)=ans(i,2);
%     beq(i+12)=ans(i+12,2);
% end
% 
% lb=[0 -Inf 0 -Inf];
% ub=[Inf Inf Inf Inf];
% 
% [x,~,e]=linprog(f,[],[],Aeq,beq,lb,ub);

for count=1:153
    
    for i=1:12
        if joint_prob(count,i)== 0
           joint_prob(count,i) = 0.000001;
        end
        
        if joint_prob(count,i+12)== 0
           joint_prob(count,i+12) = 0.000001;
        end
            
    end
    
    joint_prob(count,1:12)=joint_prob(count,1:12)/(sum(joint_prob(count,1:12)));
    joint_prob(count,13:24)=joint_prob(count,13:24)/(sum(joint_prob(count,13:24)));
    
    
    for i=1:12
        if cond_prob(count,i)== 0
          cond_prob(count,i) = 0.000001;
        end
        
        if cond_prob(count,i+12)== 0
           cond_prob(count,i+12) = 0.000001;
        end
            
    end
    
    for j=1:6
        cond_prob(count,2*(j-1)+1:2*j)=cond_prob(count,2*(j-1)+1:2*j)/(sum(cond_prob(count,2*(j-1)+1:2*j)));
        cond_prob(count,12+2*(j-1)+1:12+2*j)=cond_prob(count,12+2*(j-1)+1:12+2*j)/(sum(cond_prob(count,12+2*(j-1)+1:12+2*j)));
    end
    
    for i=1:6
        if prob_act(count,i)== 0
           prob_act(count,i) = 0.000001;
        end
        
        if prob_act(count,i+6)== 0
           prob_act(count,i+6) = 0.000001;
        end
            
    end
    
    prob_act(count,1:6)=prob_act(count,1:6)/(sum(prob_act(count,1:6)));
    prob_act(count,7:12)=prob_act(count,7:12)/(sum(prob_act(count,7:12)));
    
end


% clear all;clc;
% load('robustness_Renyi_1.mat');
% i=0;
% dev_opt=zeros(99,45,24);
% norm_dev_opt=zeros(99,45,2);
% count_1=0;
% for beta=0.01:0.01:0.99
%     count_1=count_1+1;
% arr=zeros(1,12);
% one=ones(1,1,12);
% for i=1:28
%     dev_opt(count_1,i,1:12) = (x_arr_renyi(count_1,i,1:12) + x_arr_renyi(count_1,i,36+2)*one - x_arr_renyi(count_1,i,36+1)*grad_renyi(count_1,i,1:12))/(x_arr_renyi(count_1,i,36+1)*sqrt(sum(grad_renyi(count_1,i,1:12).^2)));
%     dev_opt(count_1,i,13:24)= (x_arr_renyi(count_1,i,13:24) + x_arr_renyi(count_1,i,36+4)*one - x_arr_renyi(count_1,i,36+3)*grad_renyi(count_1,i,13:24))/(x_arr_renyi(count_1,i,36+3)*sqrt(sum(grad_renyi(count_1,i,13:24).^2)));
%     norm_dev_opt(count_1,i,1)=sqrt(sum(dev_opt(count_1,i,1:12).^2));
%     norm_dev_opt(count_1,i,2)=sqrt(sum(dev_opt(count_1,i,13:24).^2));
%     
% end
% end
% s=0;
% count=0;
% for beta=1:99
%     s=0;
%     count=0;
%     for i=1:45
%         for j=1:2
%             if norm_dev_opt(beta,i,j) ~= 0
%                 s=s+norm_dev_opt(beta,i,j);
%                 count=count+1;
%             end
%         end
%     end
%     norm_dev_beta_avg(beta)=s/count;
% end


clear all;clc;
load('robustness_Shannon_1.mat');
i=0;
dev_opt_shannon=zeros(45,24);
norm_dev_opt=zeros(45,2);
count_1=0;
%for beta=0.01:0.01:0.99
    count_1=count_1+1;
arr=zeros(1,12);
one=ones(1,12);
for i=1:45
    dev_opt_shannon(i,1:12) = (x_arr_shannon(i,1:12) + x_arr_shannon(i,36+2)*one - x_arr_shannon(i,36+1)*grad_shannon(i,1:12))/(x_arr_shannon(i,36+1)*sqrt(sum(grad_shannon(i,1:12).^2)));
    dev_opt_shannon(i,13:24)= (x_arr_shannon(i,13:24) + x_arr_shannon(i,36+4)*one - x_arr_shannon(i,36+3)*grad_shannon(i,13:24))/(x_arr_shannon(i,36+3)*sqrt(sum(grad_shannon(i,13:24).^2)));
    norm_dev_opt(i,1)=sqrt(sum(dev_opt_shannon(i,1:12).^2));
    norm_dev_opt(i,2)=sqrt(sum(dev_opt_shannon(i,13:24).^2));
    
end
%end
s=0;
count=0;
for i=1:45
    for j=1:2
        if norm_dev_opt(i,j)~=0
            s=s+norm_dev_opt(i,j);
            count=count+1;
        end
    end
end

% i=0;
% for i=1:153
%     state_prob(i,1)= sum(joint_prob(i,1:2:11));
%     state_prob(i,2)= sum(joint_prob(i,2:2:12));
%     state_prob(i,3)= sum(joint_prob(i,13:2:23)); 
%     state_prob(i,4)= sum(joint_prob(i,14:2:24));
% end
% 


