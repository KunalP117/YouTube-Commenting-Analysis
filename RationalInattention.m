clear


%Parameters of optimization problem
J = 6; % comments ({low count, negative; low count, neutral; low count, positive;
                %high count, negative; high count, neutral; high count, positive;
K = 2; % decision problems (category of video {most popular; other})
X = 2; % viewcount ({high, low})

%Construct utility for each frame
load('frame1_probability.mat')
x1 = milp_generalcost(J, K, X, cond_prob, prob_act);
%c1 = ordinal_cost(x1, prob_act, X, J, K);
load('frame2_probability.mat')
x2 = milp_generalcost(J, K, X, cond_prob, prob_act);
%c2 = ordinal_cost(x2, prob_act, X, J, K);
load('frame3_probability.mat')
x3 = milp_generalcost(J, K, X, cond_prob, prob_act);
%c3 = ordinal_cost(x3, prob_act, X, J, K);
load('frame4_probability.mat')
x4 = milp_generalcost(J, K, X, cond_prob, prob_act);
%c4 = ordinal_cost(x4, prob_act, X, J, K);

%%
% fig=figure(1);
% indx_x1 = [1+((1:6)-1)*X, 1+((1:6)-1)*X+X*J];
% indx_x2 = [2+((1:6)-1)*X, 2+((1:6)-1)*X+X*J];
% subplot(1,4,1)
% bg = bar(horzcat(x1(indx_x1),x1(indx_x2)),'FaceColor', 'flat');
% hold on
% plot([6.5 6.5],[-0.05 1.05], 'k--','LineWidth',2)
% set(bg(1),'FaceColor',[0 0 0]);
% set(bg(2),'FaceColor',[0.800000011920929 0.800000011920929 0.800000011920929]);
% ylabel('u(x,a,f=1)','fontsize',24,'interpreter','latex')
% xlabel('a','fontsize',24,'interpreter','latex')
% text(2,1.0,'x=1','fontsize',24,'interpreter','latex')
% text(9,1.0,'x=2','fontsize',24,'interpreter','latex')
% set(gca,'ytick',[])
% set(gca,'xtick',[1 2 3 4 5 6 7 8 9 10 11 12]);
% set(gca,'xticklabels',[1 2 3 4 5 6 1 2 3 4 5 6],'fontsize',22);
% xlim([0.5,12.5])
% ylim([-0.05 1.05])
% subplot(1,4,2)
% bg = bar(horzcat(x2(indx_x1),x2(indx_x2)));
% hold on
% plot([6.5 6.5],[-0.05 1.05], 'k--','LineWidth',2)
% set(bg(1),'FaceColor',[0 0 0]);
% set(bg(2),'FaceColor',[0.800000011920929 0.800000011920929 0.800000011920929]);
% ylabel('u(x,a,f=2)','fontsize',24,'interpreter','latex')
% xlabel('a','fontsize',24,'interpreter','latex')
% text(2,1.0,'x=1','fontsize',24,'interpreter','latex')
% text(9,1.0,'x=2','fontsize',24,'interpreter','latex')
% set(gca,'ytick',[])
% set(gca,'xtick',[1 2 3 4 5 6 7 8 9 10 11 12]);
% set(gca,'xticklabels',[1 2 3 4 5 6 1 2 3 4 5 6],'fontsize',22);
% xlim([0.5,12.5])
% ylim([-0.05 1.05])
% subplot(1,4,3)
% bg = bar(horzcat(x3(indx_x1),x3(indx_x2)));
% hold on
% plot([6.5 6.5],[-0.05 1.05], 'k--','LineWidth',2)
% set(bg(1),'FaceColor',[0 0 0]);
% set(bg(2),'FaceColor',[0.800000011920929 0.800000011920929 0.800000011920929]);
% ylabel('u(x,a,f=3)','fontsize',24,'interpreter','latex')
% xlabel('a','fontsize',24,'interpreter','latex')
% text(2,1.0,'x=1','fontsize',24,'interpreter','latex')
% text(9,1.0,'x=2','fontsize',24,'interpreter','latex')
% set(gca,'ytick',[])
% set(gca,'xtick',[1 2 3 4 5 6 7 8 9 10 11 12]);
% set(gca,'xticklabels',[1 2 3 4 5 6 1 2 3 4 5 6],'fontsize',22);
% xlim([0.5,12.5])
% ylim([-0.05 1.05])
% subplot(1,4,4)
% bg = bar(horzcat(x4(indx_x1),x4(indx_x2)));
% hold on
% plot([6.5 6.5],[-0.05 1.05], 'k--','LineWidth',2)
% set(bg(1),'FaceColor',[0 0 0]);
% set(bg(2),'FaceColor',[0.800000011920929 0.800000011920929 0.800000011920929]);
% ylabel('u(x,a,f=4)','fontsize',24,'interpreter','latex')
% xlabel('a','fontsize',24,'interpreter','latex')
% text(2,1.0,'x=1','fontsize',24,'interpreter','latex')
% text(9,1.0,'x=2','fontsize',24,'interpreter','latex')
% set(gca,'ytick',[])
% set(gca,'xtick',[1 2 3 4 5 6 7 8 9 10 11 12]);
% set(gca,'xticklabels',[1 2 3 4 5 6 1 2 3 4 5 6],'fontsize',22);
% xlim([0.5,12.5])
% ylim([-0.05 1.05])
% 
% fig.PaperUnits = 'inches';
% fig.PaperPosition = [0 0 16 4];
% print('utilityfunction_generalcost.jpeg','-djpeg','-r300')

% plot(x1(indx_x1), 'b-', 'linewidth', 2)
% hold on
% plot(x1(indx_x2), 'r-', 'linewidth', 2)
% plot(x2(indx_x1)+1.1, 'b-', 'linewidth', 2)
% plot(x2(indx_x2)+1.1, 'r-', 'linewidth', 2)
% hold off
% xlim([1,12])
