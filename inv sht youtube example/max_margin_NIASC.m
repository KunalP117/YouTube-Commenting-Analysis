clear; clc; close all;
load('data_ready_for_NIASC.mat');

num_states = length(unique_states);
num_actions = length(unique_actions);
num_categories = length(unique_categories);

%%%%%%%%%%%%% GENERATE MAX-MARGIN IRL REWARDS %%%%%%%%%%%
[reward_max_margin, max_margin] = compute_max_margin_NIASC(num_states,num_actions,num_categories,joint_prob_80,post_prob_80);
reward_max_margin = reward_max_margin(1:num_categories*num_states*num_actions);
reward_max_margin = reshape(reward_max_margin,num_states*num_actions,[])';


%%%% COMPUTE PREDICTED p(a|x) %%%%%%%%%%%%%%%%%%%%%%%%
cond_prob_20_est_max_margin = zeros(num_categories,num_actions*num_states);
for categ_iter = 1:num_categories
    % reward: each column is u_{\category}(*,a).
    reward = reshape(reward_max_margin(categ_iter,:),num_actions,[])';
    for state = 1:num_states
        for action = 1:num_actions
            obs_lkd = cond_prob_80(categ_iter,action:num_actions:(num_states-1)*num_actions+action);
            belief = (obs_lkd.*prior_20(categ_iter))/sum((obs_lkd.*prior_20(categ_iter)));
            [~,optimal_action] = max(belief*reward); % myopic reward
            cond_prob_20_est_max_margin(categ_iter,(state-1)*num_actions + optimal_action) = cond_prob_20_est_max_margin(categ_iter,(state-1)*num_actions + optimal_action) + cond_prob_80(categ_iter,(state-1)*num_actions + action) ;
        end
    end
end

save max_margin_prediction_80_20.mat


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [reward,margin] = compute_max_margin_NIASC(num_states,num_actions,num_categories,joint_prob_80,post_prob_80,reward_maxent)
% total variables = all rewards + opt costs + 1 (for margin)
% Rewards indexing follows the same structure as joint_prob (state 1, action 1:6, state 2, action 1:6)
num_variables = num_categories*num_states*num_actions + num_categories + 1;
lb = zeros(num_variables,1);
ub = 10*ones(num_variables,1);

% normalization: setting u(x,a) for each category to be 1 for x=1,a=1.
lb(1) = 1;
ub(1) = 1;

start = zeros(num_variables,1);


options = optimoptions('fmincon','MaxFunctionEvaluations',1e+6,'PlotFcn','optimplotfval','StepTolerance',1e-10);
% objective : maximize margin subject to NIAS, NIAC
[reward,margin] = fmincon(@(x) -x(num_variables), start,[],[],[],[],lb,ub,@nonlin,options);


function [c,ceq] = nonlin(x)
ceq = [];
c = zeros(num_categories*(num_categories-1) + num_categories*num_actions*num_actions-1,num_variables);
count = 1;
for categ_iter = 1:num_categories
    
    % reward choice has dimensions num_states x num_actions, best to
    % perform operations on this structure
    reward_choice = reshape(x((categ_iter-1)*num_states*num_actions + 1:categ_iter*num_states*num_actions,:)',num_actions,[])';
    
    % NIAS
    for act_iter = 1:num_actions
        for act_iter_two = 1:num_actions
            if act_iter ~= act_iter_two
                % NIAS inequality (less than zero format)
                c(count) = -post_prob_80(categ_iter,act_iter:num_actions:(num_states-1)*num_actions + act_iter)*(reward_choice(:,act_iter) - reward_choice(:,act_iter_two)) + x(num_variables);
                count = count + 1;
            end
        end
    end
    
    % NIAC
    for categ_iter_two = 1:num_categories
        if categ_iter ~= categ_iter_two
            % expected reward for categ_iter
            c(count) = - sum(x((categ_iter-1)*num_states*num_actions + 1: categ_iter*num_states*num_actions)'.*joint_prob_80(categ_iter,:));
            % max expected reward for categ_iter_two
            candidate_two_reward = sum( max(reshape(joint_prob_80(categ_iter_two,:),num_actions,[])*reward_choice,[],2) );
            
            % NIAC inequality (neg. of (reward - cost - reward_2 + cost_2 +
            % margin) less than 0)
            c(count) = c(count) + x(num_categories*num_actions*num_states + categ_iter) + candidate_two_reward - x(num_categories*num_actions*num_states + categ_iter_two) + x(num_variables); 
            count = count + 1;
        end
    end
    
    
end
end

end