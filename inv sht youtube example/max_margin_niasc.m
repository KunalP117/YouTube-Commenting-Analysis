clear; clc; close all;

%%% Code the generates this file is 'test_train_80_20.m'
load('../80_percent_data.mat');

unique_categories = unique(category);
unique_states = unique(viewcount_disc);
unique_actions = unique(actions);

cond_prob_80 =zeros(length(unique_categories),length(unique_states)*length(unique_actions));
prior_80 = zeros(length(unique_categories),length(unique_states));
act_prob_80 = zeros(length(unique_categories), length(unique_actions));
post_prob_80 = zeros(length(unique_categories),length(unique_states)*length(unique_actions));
joint_prob_80 = zeros(length(unique_categories),length(unique_states)*length(unique_actions));

for categ_iter = 1:length(unique_categories)
    c1 = find(category == unique_categories(categ_iter));
    state_restricted = viewcount_disc(c1);
    actions_restricted = actions(c1);
    
    for state_iter = 1:length(unique_states)
        c2 = find(state_restricted == unique_states(state_iter));
        prior_80(categ_iter,state_iter) = length(c2)/length(state_restricted); 
        
        actions_double_restricted = actions_restricted(c2);
        for action_iter = 1:length(unique_actions)
            c3_act = find(actions_restricted == unique_actions(action_iter));
            act_prob_80(categ_iter,action_iter) = length(c3_act)/length(actions_restricted);
            
            c3_joint = find(actions_double_restricted == unique_actions(action_iter));
            joint_prob_80(categ_iter,(state_iter-1)*length(unique_actions) + action_iter) = length(c3_joint)/length(actions_restricted);            
            cond_prob_80(categ_iter,(state_iter-1)*length(unique_actions) + action_iter) = joint_prob_80(categ_iter, (state_iter-1)*length(unique_actions) + action_iter)/prior_80(categ_iter,state_iter);
            post_prob_80(categ_iter,(state_iter-1)*length(unique_actions) + action_iter) = joint_prob_80(categ_iter, (state_iter-1)*length(unique_actions) + action_iter)/act_prob_80(categ_iter,action_iter);
        end
        
        % cond_prob has elements [ p(a1|x1) p(a2|x1) p(a3|x1)  .. p(a6|x1) p(a1|x2) p(a2|x2) .. p(a6|x2) ]

    end
end

% CONVERT NaNs to zeros
joint_prob_80(isnan(joint_prob_80)) = 0;
post_prob_80(isnan(post_prob_80)) = 0;
cond_prob_80(isnan(cond_prob_80)) = 0;
act_prob_80(isnan(act_prob_80)) = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Code the generates this file is 'test_train_80_20.m'
load('../20_percent_data.mat');

unique_categories = unique(category);
unique_states = unique(viewcount_disc);
unique_actions = unique(actions);

cond_prob_20 =zeros(length(unique_categories),length(unique_states)*length(unique_actions));
prior_20 = zeros(length(unique_categories),length(unique_states));
act_prob_20 = zeros(length(unique_categories), length(unique_actions));
post_prob_20 = zeros(length(unique_categories),length(unique_states)*length(unique_actions));
joint_prob_20 = zeros(length(unique_categories),length(unique_states)*length(unique_actions));

for categ_iter = 1:length(unique_categories)
    c1 = find(category == unique_categories(categ_iter));
    state_restricted = viewcount_disc(c1);
    actions_restricted = actions(c1);
    
    for state_iter = 1:length(unique_states)
        c2 = find(state_restricted == unique_states(state_iter));
        prior_20(categ_iter,state_iter) = length(c2)/length(state_restricted); 
        
        actions_double_restricted = actions_restricted(c2);
        for action_iter = 1:length(unique_actions)
            c3_act = find(actions_restricted == unique_actions(action_iter));
            act_prob_20(categ_iter,action_iter) = length(c3_act)/length(actions_restricted);
            
            c3_joint = find(actions_double_restricted == unique_actions(action_iter));
            joint_prob_20(categ_iter,(state_iter-1)*length(unique_actions) + action_iter) = length(c3_joint)/length(actions_restricted);            
            cond_prob_20(categ_iter,(state_iter-1)*length(unique_actions) + action_iter) = joint_prob_20(categ_iter, (state_iter-1)*length(unique_actions) + action_iter)/prior_20(categ_iter,state_iter);
            post_prob_20(categ_iter,(state_iter-1)*length(unique_actions) + action_iter) = joint_prob_20(categ_iter, (state_iter-1)*length(unique_actions) + action_iter)/act_prob_20(categ_iter,action_iter);
        end
        
        % cond_prob has elements [ p(a1|x1) p(a2|x1) p(a3|x1)  .. p(a6|x1) p(a1|x2) p(a2|x2) .. p(a6|x2) ]

    end
end

% CONVERT NaNs to zeros
joint_prob_20(isnan(joint_prob_20)) = 0;
post_prob_20(isnan(post_prob_20)) = 0;
cond_prob_20(isnan(cond_prob_20)) = 0;
act_prob_20(isnan(act_prob_20)) = 0;

save('data_ready_for_NIASC.mat');