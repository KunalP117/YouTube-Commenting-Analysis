clear; clc; close all;

%%% Code the generates this file is 'test_train_80_20.m'
load('../80_percent_data.mat');

unique_categories = unique(category);
unique_states = unique(viewcount_disc);
unique_actions = unique(actions);

cond_prob_80 =zeros(length(unique_categories),length(unique_states)*length(unique_actions));
prior_80 = zeros(length(unique_categories),length(unique_states));

for categ_iter = 1:length(unique_categories)
    c1 = find(category == unique_categories(categ_iter));
    for state_iter = 1:length(unique_states)
        c2 = find(viewcount_disc(c1) == unique_states(state_iter));
        prior_80(categ_iter,state_iter) = length(c2)/length(c1);
        for action_iter = 1:length(unique_actions)
            c3 = find(actions(c2) == unique_actions(action_iter));
            cond_prob_80(categ_iter, (state_iter-1)*length(unique_actions) + action_iter) = length(c3)/length(c2);
        end
        
        % cond_prob has elements [ p(a1|x1) p(a2|x1) p(a3|x1)  .. p(a6|x1) p(a1|x2) p(a2|x2) .. p(a6|x2) ]

    end
end

load('../20_percent_data.mat');
cond_prob_20 =zeros(length(unique_categories),length(unique_states)*length(unique_actions));
prior_20 = zeros(length(unique_categories),length(unique_states));

for categ_iter = 1:length(unique_categories)
    c1 = find(category == unique_categories(categ_iter));
    for state_iter = 1:length(unique_states)
        prior_20(categ_iter,state_iter) = length(c2)/length(c1);
        c2 = find(viewcount_disc(c1) == unique_states(state_iter));
        for action_iter = 1:length(unique_actions)
            c3 = find(actions(c2) == unique_actions(action_iter));
            cond_prob_20(categ_iter, (state_iter-1)*length(unique_actions) + action_iter) = length(c3)/length(c2);
        end
        
        % cond_prob has elements [ p(a1|x1) p(a2|x1) p(a3|x1)  .. p(a6|x1) p(a1|x2) p(a2|x2) .. p(a6|x2) ]

    end
end
