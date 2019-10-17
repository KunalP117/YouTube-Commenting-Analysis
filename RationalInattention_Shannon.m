clear
clc


%Parameters of optimization problem
J = 6; % comments ({low count, negative; low count, neutral; low count, positive;
                %high count, negative; high count, neutral; high count, positive;
K = 2; % decision problems (category of video {most popular; other})
X = 2; % viewcount ({high, low})

%Construct utility for each frame
% load('frame1_probability.mat')
% x1 = milp_generalcost(J, K, X, cond_prob, prob_act);
% load('frame2_probability.mat')
% x2 = milp_generalcost(J, K, X, cond_prob, prob_act);
% load('frame3_probability.mat')
% x3 = milp_generalcost(J, K, X, cond_prob, prob_act);
% load('frame4_probability.mat')
% x4 = milp_generalcost(J, K, X, cond_prob, prob_act);

load('frame1_probability.mat')
state_prob=horzcat(state_prob,state_prob);
[x1,e1] = milp_shannoncost(J, K, X, cond_prob, prob_act);
load('frame2_probability.mat')
state_prob=horzcat(state_prob,state_prob);
[x2,e2] = milp_shannoncost(J, K, X, cond_prob, prob_act);
load('frame3_probability.mat')
state_prob=horzcat(state_prob,state_prob);
[x3,e3] = milp_shannoncost(J, K, X, cond_prob, prob_act);
load('frame4_probability.mat')
state_prob=horzcat(state_prob,state_prob);
[x4,e4] = milp_shannoncost(J, K, X, cond_prob, prob_act);

%%
% fig=figure(1);
% indx_x1 = [1+((1:6)-1)*X, 1+((1:6)-1)*X+X*J];
% indx_x2 = [2+((1:6)-1)*X, 2+((1:6)-1)*X+X*J];
% bg = bar(horzcat(x4(indx_x1),x4(indx_x2)),'FaceColor', 'flat');
% hold on
% plot([6.5 6.5],[-0.05 1.05], 'k--')
% set(bg(1),'FaceColor',[0 0 0]);
% set(bg(2),'FaceColor',[0.800000011920929 0.800000011920929 0.800000011920929]);
% ylabel('u(x,a,f=4)','fontsize',14,'interpreter','latex')
% xlabel('a','fontsize',14,'interpreter','latex')
% text(2,1.0,'x=1','fontsize',14,'interpreter','latex')
% text(9,1.0,'x=2','fontsize',14,'interpreter','latex')
% set(gca,'ytick',[])
% set(gca,'xtick',[1 2 3 4 5 6 7 8 9 10 11 12]);
% set(gca,'xticklabels',[1 2 3 4 5 6 1 2 3 4 5 6]);
% xlim([0.5,12.5])
% ylim([-0.05 1.05])
% fig.PaperUnits = 'inches';
% fig.PaperPosition = [0 0 5 4];
% print('utilityfunction_shannoncost.jpeg','-djpeg','-r300')


