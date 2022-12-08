clear; clc; close all;
load('data_ready_for_NIASC.mat');


num_states = length(unique_states);
num_actions = length(unique_actions);
num_categories = length(unique_categories);

%%%%%%%%%%%%% GENERATE MAX-ENT IRL REWARDS %%%%%%%%%%%
tolerance = 1e-6;
neg_utility = -3;
% if prob < tolerance, u(x,a) = neg utility,
% else, u(x,a) = ln(p(x|a)/min_elem),
% where min_elem is the smallest posterior probability
% p(x|a) that is greater than tolerance.
reward_maxent = zeros(num_categories,num_states*num_actions);

for categ_iter = 1:num_categories
    post_prob = post_prob_80(categ_iter,:);
    min_prob = min(post_prob(post_prob>tolerance));
    for iter = 1:num_states*num_actions
        if post_prob_80(categ_iter,iter)<= tolerance
            reward_maxent(categ_iter,iter) = neg_utility;
        else
            reward_maxent(categ_iter,iter) = log(post_prob_80(categ_iter,iter)/min_prob);
        end

    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%

cond_prob_20_est = zeros(num_categories,num_actions*num_states);
for categ_iter = 1:num_categories
    % reward: each column is u_{\category}(*,a).
    reward = reshape(reward_maxent(categ_iter,:),num_actions,[])';
    for state = 1:num_states
        for action = 1:num_actions
            obs_lkd = cond_prob_80(categ_iter,action:num_actions:(num_states-1)*num_actions+action);
            belief = (obs_lkd.*prior_20(categ_iter))/sum((obs_lkd.*prior_20(categ_iter)));
            [~,optimal_action] = max(belief*reward); % myopic reward
            cond_prob_20_est(categ_iter,(state-1)*num_actions + optimal_action) = cond_prob_20_est(categ_iter,(state-1)*num_actions + optimal_action) + cond_prob_80(categ_iter,(state-1)*num_actions + action) ;
        end
    end
end
% COMPUTE KL DIVERGENCE BETWEEN cond_prob_20_est and cond_prob_20

%cond_prob_20_est(cond_prob_20_est<tolerance) = tolerance; % to eliminate zeros in denominator
cross_log_cond_prob_20 = log(cond_prob_20)./log(cond_prob_20_est);
cross_log_cond_prob_20(isinf(cross_log_cond_prob_20)) = 0;
cross_log_cond_prob_20(isnan(cross_log_cond_prob_20)) = 0;

cross_log_cond_prob_20_est = log(cond_prob_20_est)./log(cond_prob_20);
cross_log_cond_prob_20_est(isinf(cross_log_cond_prob_20_est)) = 0;
cross_log_cond_prob_20_est(isnan(cross_log_cond_prob_20_est)) = 0;


KLdiv_mat = (cross_log_cond_prob_20).*cond_prob_20;
avg_KLdiv_mat_category = sum(KLdiv_mat,2)/num_states;

% TWO SIDED KL-DIVERGENCE
two_sided_KLdiv_mat = 0.5*( (cross_log_cond_prob_20_est).*cond_prob_20_est + (cross_log_cond_prob_20_est).*cond_prob_20_est );
avg_two_sided_KLdiv_mat_category = sum(two_sided_KLdiv_mat,2)/num_states;
% figure();
% stem(avg_KLdiv_mat_category);

% figure();
% ecdf(avg_KLdiv_mat_category);
% 
% title('KL-Divergence');
% 
% figure();
% ecdf(avg_two_sided_KLdiv_mat_category);
% title('Symmetric KL divergence');

% % COMPUTE JENSEN-SHANNON DIVERGENCE (guaranteed to be in [0,1])
% cond_prob_20_avg_est_true = 0.5*(cond_prob_20_est + cond_prob_20);
% 
% % compute cross log of true with (true + est)/2
% cross_log_cond_prob_20_true = log(cond_prob_20)./log(cond_prob_20_avg_est_true);
% cross_log_cond_prob_20_true(isinf(cross_log_cond_prob_20_true)) = 0;
% cross_log_cond_prob_20_true(isnan(cross_log_cond_prob_20_true)) = 0;
% 
% % compute cross log of est with (true + est)/2
% cross_log_cond_prob_20_est = log(cond_prob_20_est)./log(cond_prob_20_avg_est_true);
% cross_log_cond_prob_20_est(isinf(cross_log_cond_prob_20_est)) = 0;
% cross_log_cond_prob_20_est(isnan(cross_log_cond_prob_20_est)) = 0;
% 
% 
% 
% Jensen_Shannon_div_mat = 0.5*((cross_log_cond_prob_20_true.*cond_prob_20) + (cross_log_cond_prob_20_est.*cond_prob_20_est));
% avg_Jensen_Shannon_div_category = sum(Jensen_Shannon_div_mat,2)/num_states;
% 
% % figure();
% % stem(avg_Jensen_Shannon_div_category);
% 
% figure();
% ecdf(avg_Jensen_Shannon_div_category);
% title('Jensen-Shannon Divergence');

%%%%%%%%%%%%% Chi-Squared Distance %%%%%%%%%
inv_diff_cond_probs = ones(num_categories,num_states*num_actions)./(cond_prob_20 + cond_prob_20_est);
inv_diff_cond_probs(isinf(inv_diff_cond_probs)) = 0;
inv_diff_cond_probs(isnan(inv_diff_cond_probs)) = 0;


chi_squared_distance = (0.5/num_states)*sum( ((cond_prob_20 - cond_prob_20_est).^2).*inv_diff_cond_probs, 2);

% figure();
% stem(chi_squared_distance);

% figure();
% ecdf(chi_squared_distance);
% title('Chi-Squared Distance');

%%%%%%%% TOTAL VARIATION DISTANCE %%%%%%%%%%%%%%

tvdistance_mat = 0.5*sum(abs(cond_prob_20_est - cond_prob_20),2)/num_states;

% figure();
% ecdf(tvdistance_mat);
% title('TV distance');

figure();
subplot(1,2,1);
ecdf(chi_squared_distance);
title('Chi-Squared Distance');

subplot(1,2,2);
ecdf(tvdistance_mat);
title('Total Variation Distance');



save max_ent_prediction_80_20.mat


