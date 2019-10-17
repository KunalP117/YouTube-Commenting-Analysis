% clear all;
% clc;
% load('n4_embed_data.mat');
% %%% a lot of processed data (content of this code prior to calculating prob. vectors) is in temp_data.mat
% clus_dat=transpose(lat_rep);
% c1=0;c2=0;c3=0;c4=0;
% c1_non=0;c2_non=0;c3_non=0;c4_non=0;
% f1_dat=zeros(2,10);
% f2_dat=zeros(2,10);
% f3_dat=zeros(2,10);
% f4_dat=zeros(2,10);
% 
% f1_act=zeros(1,10);
% f2_act=zeros(1,10);                                                         
% f3_act=zeros(1,10);
% f4_act=zeros(1,10);
% 
% f1_view=zeros(1,10);
% f2_view=zeros(1,10);
% f3_view=zeros(1,10);
% f4_view=zeros(1,10);
% 
% f1_dat_non=zeros(2,10);
% f2_dat_non=zeros(2,10);
% f3_dat_non=zeros(2,10);
% f4_dat_non=zeros(2,10);
% 
% f1_act_non=zeros(1,10);
% f2_act_non=zeros(1,10);
% f3_act_non=zeros(1,10);
% f4_act_non=zeros(1,10);
% 
% f1_view_non=zeros(1,10);
% f2_view_non=zeros(1,10);
% f3_view_non=zeros(1,10);
% f4_view_non=zeros(1,10);
% 
% a=0;
% m=unique(category);
% count=zeros(1,max(m));
% for i=1:114528
%     a=category(i);
%     count(a)=count(a)+1;
% end
% 
% [a max_ind]=max(count);
% 
% 
% 
% for i=1:114528
%     if frame(i)==0
%         if category(i)==max_ind
%             c1=c1+1;
%             f1_dat(:,c1)=lat_rep(:,i);
%             f1_act(c1)=actions(i);
%             f1_view(c1)=viewcount(i);
%         else
%             c1_non=c1_non+1;
%             f1_dat_non(:,c1_non)=lat_rep(:,i);
%             f1_act_non(c1_non)=actions(i);
%             f1_view_non(c1_non)=viewcount(i);            
%         end
%     elseif frame(i)==1
%         if category(i)==max_ind
%             c2=c2+1;
%             f2_dat(:,c2)=lat_rep(:,i);
%             f2_act(c2)=actions(i);
%             f2_view(c2)=viewcount(i);
%         else
%             c2_non=c2_non+1;
%             f2_dat_non(:,c2_non)=lat_rep(:,i);
%             f2_act_non(c2_non)=actions(i);
%             f2_view_non(c2_non)=viewcount(i);            
%         end
%     elseif frame(i)==2
%         if category(i)==max_ind
%             c3=c3+1;
%             f3_dat(:,c3)=lat_rep(:,i);
%             f3_act(c3)=actions(i);
%             f3_view(c3)=viewcount(i);
%         else
%             c3_non=c3_non+1;
%             f3_dat_non(:,c3_non)=lat_rep(:,i);
%             f3_act_non(c3_non)=actions(i);
%             f3_view_non(c3_non)=viewcount(i);            
%         end
%     else
%         if category(i)==max_ind
%             c4=c4+1;
%             f4_dat(:,c4)=lat_rep(:,i);
%             f4_act(c4)=actions(i);
%             f4_view(c4)=viewcount(i);
%         else
%             c4_non=c4_non+1;
%             f4_dat_non(:,c4_non)=lat_rep(:,i);
%             f4_act_non(c4_non)=actions(i);
%             f4_view_non(c4_non)=viewcount(i);            
%         end
%     end
% end
% 
% %%%%%%
% %Perform k-means clustering for each frame and each decision problem
%  cl_MAX=15;
% %storing the indices of each video in each category of each frame
% l1=0;l1_non=0;
% l2=0;l2_non=0;
% l3=0;l3_non=0;
% l4=0;l4_non=0;
% 
% %storing the cluster head of each video in each category of each frame
% k_cl1=0;k_cl1_non=0;
% k_cl2=0;k_cl2_non=0;
% k_cl3=0;k_cl3_non=0;
% k_cl4=0;k_cl4_non=0;
% 
% [l1,k_cl1]=kmeans(transpose(f1_dat),cl_MAX,'MaxIter',10000);
% [l2,k_cl2]=kmeans(transpose(f2_dat),cl_MAX,'MaxIter',10000);
% [l3,k_cl3]=kmeans(transpose(f3_dat),cl_MAX,'MaxIter',10000);
% [l4,k_cl4]=kmeans(transpose(f4_dat),cl_MAX,'MaxIter',10000);
% 
% [l1_non,k_cl1_non]=kmeans(transpose(f1_dat_non),cl_MAX,'MaxIter',10000);
% [l2_non,k_cl2_non]=kmeans(transpose(f2_dat_non),cl_MAX,'MaxIter',10000);
% [l3_non,k_cl3_non]=kmeans(transpose(f3_dat_non),cl_MAX,'MaxIter',10000);
% [l4_non,k_cl4_non]=kmeans(transpose(f4_dat_non),cl_MAX,'MaxIter',10000);
% 
% 
% 
% dist_arr=zeros(cl_MAX,cl_MAX);
% i=0;j=0;
% for i=1:cl_MAX
%     for j=1:cl_MAX
%         dist_arr(i,j)=sum((k_cl1(i)-k_cl1_non(j)).^2);
%     end
% end
% 
% min_dist_1=zeros(1,cl_MAX);
% min_dist_2=zeros(1,cl_MAX);
% a=0;
% for i=1:cl_MAX
%     [a,min_dist_1(i)]=min(dist_arr(i,:));
%     [a,min_dist_2(i)]=min(dist_arr(:,i));
% end
% 
% % 
% % [alpha,beta,gamma] = unique(min_dist_arr);
% % 
% % %write genetic algorithm for best video matches in both decision problems
% % lb=zeros(1,cl_MAX*cl_MAX);
% % ub=ones(1,cl_MAX*cl_MAX);
% % nvars=cl_MAX*cl_MAX;
% % one_arr=ones(1,cl_MAX);
% % col_sum_arr=zeros(1,cl_MAX*cl_MAX);
% % 
% % for i=1:cl_MAX
% %     col_sum_arr(1,i*cl_MAX) = 1;
% % end
% % 
% % A=zeros(4*cl_MAX,cl_MAX*cl_MAX);
% % b=ones(4*cl_MAX,1);
% % b(cl_MAX+1:2*cl_MAX)=(-1)*ones(1,cl_MAX);
% % b(3*cl_MAX+1:4*cl_MAX)=(-1)*ones(1,cl_MAX);
% % for i=1:cl_MAX
% %     A(i,(i-1)*cl_MAX+1:i*cl_MAX)=one_arr;
% %     A(i+cl_MAX,(i-1)*cl_MAX+1:i*cl_MAX)=(-1)*one_arr;
% %     A(i+2*cl_MAX,:)= circshift(col_sum_arr,i,2);
% %     A(i+3*cl_MAX,:)=(-1)*A(i+2*cl_MAX,:);
% % end
% % IntCon=[1:nvars];
% % %%
% % k_cl_aug=zeros(2,100);
% % target_aug=zeros(2,100);
% % for i=1:cl_MAX-1
% %     k_cl_aug(:,(i-1)*cl_MAX+1:i*cl_MAX)=transpose(k_cl1);
% %     target_aug(:,(i-1)*cl_MAX+1:i*cl_MAX)=transpose(k_cl1_non);
% % end
% % fun1= @(x) sum(((k_cl_aug(1,:).*x)-target_aug(1,:)).^2) + sum(((k_cl_aug(2,:).*x)-target_aug(2,:)).^2) ;
% % %%
% % k_cl_aug=zeros(2,100);
% % target_aug=zeros(2,100);
% % for i=1:cl_MAX-1
% %     k_cl_aug(:,(i-1)*cl_MAX+1:i*cl_MAX)=transpose(k_cl2);
% %     target_aug(:,(i-1)*cl_MAX+1:i*cl_MAX)=transpose(k_cl2_non);
% % end
% % fun2= @(x) sum(((k_cl_aug(1,:).*x)-target_aug(1,:)).^2) + sum(((k_cl_aug(2,:).*x)-target_aug(2,:)).^2) ;
% % %%
% % k_cl_aug=zeros(2,100);
% % target_aug=zeros(2,100);
% % for i=1:cl_MAX-1
% %     k_cl_aug(:,(i-1)*cl_MAX+1:i*cl_MAX)=transpose(k_cl3);
% %     target_aug(:,(i-1)*cl_MAX+1:i*cl_MAX)=transpose(k_cl3_non);
% % end
% % fun3= @(x) sum(((k_cl_aug(1,:).*x)-target_aug(1,:)).^2) + sum(((k_cl_aug(2,:).*x)-target_aug(2,:)).^2) ;
% % %%
% % k_cl_aug=zeros(2,100);
% % target_aug=zeros(2,100);
% % for i=1:cl_MAX-1
% %     k_cl_aug(:,(i-1)*cl_MAX+1:i*cl_MAX)=transpose(k_cl4);
% %     target_aug(:,(i-1)*cl_MAX+1:i*cl_MAX)=transpose(k_cl4_non);
% % end
% % fun4= @(x) sum(((k_cl_aug(1,:).*x)-target_aug(1,:)).^2) + sum(((k_cl_aug(2,:).*x)-target_aug(2,:)).^2) ;
% % %%
% % 
% % ans1 = ga(fun1,nvars,A,b,[],[],lb,ub,[],IntCon);
% % ans2 = ga(fun2,nvars,A,b,[],[],lb,ub,[],IntCon);
% % ans3 = ga(fun3,nvars,A,b,[],[],lb,ub,[],IntCon);
% % ans4 = ga(fun4,nvars,A,b,[],[],lb,ub,[],IntCon);
% 
% 
% %index_set=gpuArray([1:cl_MAX]);
% %PERM_set=perms(index);
% loss=0;
% perm_set=zeros(1,10);
% perm_set=randperm(cl_MAX);
% for j=1:cl_MAX
%     loss=loss+sum((k_cl1(j,:)-k_cl1_non(perm_set(j),:)).^2);
% end
% min_match_1=zeros(1,cl_MAX);
% min_match_1=perm_set;
% loss=loss/cl_MAX;
% 
% iter=1;
% min_loss=loss;
% %min_match_1=zeros(1,10);
% MAX_iter=1000000;
% loss_arr=zeros(1,1000000);
% loss_arr(i)=loss;
% 
% for iter=2:MAX_iter
%     iter
%     perm_set=randperm(cl_MAX);
%     loss=0;
%     for j=1:cl_MAX
%         loss=loss+sum((k_cl1(j,:)-k_cl1_non(perm_set(j),:)).^2);
%     end
%     loss=loss/cl_MAX;
%     loss_arr(iter)=loss;
%     if loss <= min_loss
%         min_match_1 = perm_set;
%     end
% end
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% perm_set=zeros(1,10);
% perm_set=randperm(cl_MAX);
% for j=1:cl_MAX
%     loss=loss+sum((k_cl1(j,:)-k_cl1_non(perm_set(j),:)).^2);
% end
% min_match_2=zeros(1,cl_MAX);
% min_match_2=perm_set;
% loss=loss/cl_MAX;
% 
% iter=1;
% min_loss=loss;
% %min_match_1=zeros(1,10);
% MAX_iter=1000000;
% loss_arr=zeros(1,1000000);
% loss_arr(i)=loss;
% 
% for iter=1:MAX_iter
%     iter
%     perm_set=randperm(cl_MAX);
%     loss=0;
%     for j=1:cl_MAX
%         loss=loss+sum((k_cl2(j,:)-k_cl2_non(perm_set(j),:)).^2);
%     end
%     loss=loss/cl_MAX;
%     loss_arr(iter)=loss;
%     if loss <= min_loss
%         min_match_2 = perm_set;
%     end
% end
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% perm_set=zeros(1,10);
% perm_set=randperm(cl_MAX);
% for j=1:cl_MAX
%     loss=loss+sum((k_cl1(j,:)-k_cl1_non(perm_set(j),:)).^2);
% end
% min_match_3=zeros(1,cl_MAX);
% min_match_3=perm_set;
% loss=loss/cl_MAX;
% 
% iter=1;
% min_loss=loss;
% %min_match_1=zeros(1,10);
% MAX_iter=1000000;
% loss_arr=zeros(1,1000000);
% loss_arr(i)=loss;
% 
% for iter=1:MAX_iter
%     iter
%     perm_set=randperm(cl_MAX);
%     loss=0;
%     for j=1:cl_MAX
%         loss=loss+sum((k_cl3(j,:)-k_cl3_non(perm_set(j),:)).^2);
%     end
%     loss=loss/cl_MAX;
%     loss_arr(iter)=loss;
%     if loss <= min_loss
%         min_match_3 = perm_set;
%     end
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% perm_set=zeros(1,10);
% perm_set=randperm(cl_MAX);
% for j=1:cl_MAX
%     loss=loss+sum((k_cl1(j,:)-k_cl1_non(perm_set(j),:)).^2);
% end
% min_match_4=zeros(1,cl_MAX);
% min_match_4=perm_set;
% loss=loss/cl_MAX;
% 
% iter=1;
% min_loss=loss;
% %min_match_1=zeros(1,10);
% MAX_iter=1000000;
% loss_arr=zeros(1,1000000);
% loss_arr(i)=loss;
% 
% for iter=1:MAX_iter
%     iter
%     perm_set=randperm(cl_MAX);
%     loss=0;
%     for j=1:cl_MAX
%         loss=loss+sum((k_cl4(j,:)-k_cl4_non(perm_set(j),:)).^2);
%     end
%     loss=loss/cl_MAX;
%     loss_arr(iter)=loss;
%     if loss <= min_loss
%         min_match_3 = perm_set;
%     end
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Start processing video view counts
% view_thresh=10000;
% view_count=zeros(1,16);
% 
% 
% m=size(f1_view);m=m(2);
% for i=1:m
%   if f1_view(i) >=view_thresh
%       f1_view(i)=1;
%       view_count(1)=view_count(1)+1;
%   else
%       f1_view(i)=0;
%       view_count(2)=view_count(2)+1;
%   end
% end
% 
% m=size(f2_view);m=m(2);
% for i=1:m
%   if f2_view(i) >=view_thresh
%       f2_view(i)=1;
%       view_count(3)=view_count(3)+1;
%   else
%       f2_view(i)=0;
%       view_count(4)=view_count(4)+1;
%   end
% end
% 
% m=size(f3_view);m=m(2);
% for i=1:m
%   if f3_view(i) >=view_thresh
%       f3_view(i)=1;
%       view_count(5)=view_count(5)+1;
%   else
%       f3_view(i)=0;
%       view_count(6)=view_count(6)+1;
%   end
% end
% 
% m=size(f4_view);m=m(2);
% for i=1:m
%   if f4_view(i) >=view_thresh
%       f4_view(i)=1;
%       view_count(7)=view_count(7)+1;
%   else
%       f4_view(i)=0;
%       view_count(8)=view_count(8)+1;
%   end
% end
% 
% %%%%%%%%% non gaming %%%%%%%%
% m=size(f1_view_non);m=m(2);
% for i=1:m
%   if f1_view_non(i) >=view_thresh
%       f1_view_non(i)=1;
%       view_count(9)=view_count(9)+1;
%   else
%       f1_view_non(i)=0;
%       view_count(10)=view_count(10)+1;
%   end
% end
% 
% m=size(f2_view_non);m=m(2);
% for i=1:m
%   if f2_view_non(i) >=view_thresh
%       f2_view_non(i)=1;
%       view_count(11)=view_count(11)+1;
%   else
%       f2_view_non(i)=0;
%       view_count(12)=view_count(12)+1;
%   end
% end
% 
% m=size(f3_view_non);m=m(2);
% for i=1:m
%   if f3_view_non(i) >=view_thresh
%       f3_view_non(i)=1;
%       view_count(13)=view_count(13)+1;
%   else
%       f3_view_non(i)=0;
%       view_count(14)=view_count(14)+1;
%   end
% end
% 
% m=size(f4_view_non);m=m(2);
% for i=1:m
%   if f4_view_non(i) >=view_thresh
%       f4_view_non(i)=1;
%       view_count(15)=view_count(15)+1;
%   else
%       f4_view_non(i)=0;
%       view_count(16)=view_count(16)+1;
%   end
% end
% 

% %%%%%%%%%%%%%%%%%%%%% create data %%%%%%%%%%%%%%%%%%%%%%
%%for frame 1
cond_prob=zeros(cl_MAX,24); % p(x|a)
%joint_prob=zeros(cl_MAX,24); % p(x,a)
prob_act=zeros(cl_MAX,12); % p(a)
state_prob=zeros(1,2); 

m=size(l1); m=m(1);
m_1=size(l1_non);m_1=m_1(2);
tot=size(f1_view)+size(f1_view_non);
tot=tot(2);
state_prob(2)= (sum(f1_view)+sum(f1_view_non)) / tot;  
state_prob(1)= 1-state_prob(2);
ind_set=0;view_set=0;
ind_set_1=0;view_set_1=0;
j=1;
k=1;  
for k=1:cl_MAX
  ind_set= find(l1==k);
  ind_set_1=find(l1_non==min_match_1(k));
  m=size(ind_set);m=m(1);
  m1=size(ind_set_1);m1=m1(1);
  act_set=f1_act(ind_set);
  view_set=f1_view(ind_set);
  act_set_1=f1_act_non(ind_set_1);
  view_set_1=f1_view_non(ind_set_1);
  for i=1:6
      c=find(act_set==i);
      cond_prob(k,2+(i-1)*2)=sum(view_set)/m;
      cond_prob(k,1+(i-1)*2)=1-cond_prob(k,2+(i-1)*2);
      c=size(c);c=c(2);
      prob_act(k,i)=c/m;
  
      c=find(act_set_1==i);
      cond_prob(k,12+2+(i-1)*2)=sum(view_set_1)/m1;
      cond_prob(k,12+1+(i-1)*2)=1-cond_prob(k,12+2+(i-1)*2);
      c=size(c);c=c(2);
      prob_act(k,i+6)=c/m1;
  end
end


%%% for frame 2
cond_prob=zeros(cl_MAX,24); % p(x|a)
%joint_prob=zeros(cl_MAX,24); % p(x,a)
prob_act=zeros(cl_MAX,12); % p(a)
state_prob=zeros(1,2); 

m=size(l2); m=m(1);
m_1=size(l2_non);m_1=m_1(2);
tot=size(f2_view)+size(f2_view_non);
tot=tot(2);
state_prob(2)= (sum(f2_view)+sum(f2_view_non)) / tot;  
state_prob(1)= 1-state_prob(2);
ind_set=0;view_set=0;
ind_set_1=0;view_set_1=0;
j=1;
k=1;  
for k=1:cl_MAX
  ind_set= find(l2==k);
  ind_set_1=find(l2_non==min_match_2(k));
  m=size(ind_set);m=m(1);
  m1=size(ind_set_1);m1=m1(1);
  act_set=f2_act(ind_set);
  view_set=f2_view(ind_set);
  act_set_1=f2_act_non(ind_set_1);
  view_set_1=f2_view_non(ind_set_1);
  for i=1:6
      c=find(act_set==i);
      cond_prob(k,2+(i-1)*2)=sum(view_set)/m;
      cond_prob(k,1+(i-1)*2)=1-cond_prob(k,2+(i-1)*2);
      c=size(c);c=c(2);
      prob_act(k,i)=c/m;
  
      c=find(act_set_1==i);
      cond_prob(k,12+2+(i-1)*2)=sum(view_set_1)/m1;
      cond_prob(k,12+1+(i-1)*2)=1-cond_prob(k,12+2+(i-1)*2);
      c=size(c);c=c(2);
      prob_act(k,i+6)=c/m1;
  end
end

% 
% % %%% for frame 3
cond_prob=zeros(cl_MAX,24); % p(x|a)
%joint_prob=zeros(cl_MAX,24); % p(x,a)
prob_act=zeros(cl_MAX,12); % p(a)
state_prob=zeros(1,2); 

m=size(l3); m=m(1);
m_1=size(l3_non);m_1=m_1(2);
tot=size(f3_view)+size(f3_view_non);
tot=tot(2);
state_prob(2)= (sum(f3_view)+sum(f3_view_non)) / tot;  
state_prob(1)= 1-state_prob(2);
ind_set=0;view_set=0;
ind_set_1=0;view_set_1=0;
j=1;
k=1;  
for k=1:cl_MAX
  ind_set= find(l3==k);
  ind_set_1=find(l3_non==min_match_3(k));
  m=size(ind_set);m=m(1);
  m1=size(ind_set_1);m1=m1(1);
  act_set=f3_act(ind_set);
  view_set=f3_view(ind_set);
  act_set_1=f3_act_non(ind_set_1);
  view_set_1=f3_view_non(ind_set_1);
  for i=1:6
      c=find(act_set==i);
      cond_prob(k,2+(i-1)*2)=sum(view_set)/m;
      cond_prob(k,1+(i-1)*2)=1-cond_prob(k,2+(i-1)*2);
      c=size(c);c=c(2);
      prob_act(k,i)=c/m;
  
      c=find(act_set_1==i);
      cond_prob(k,12+2+(i-1)*2)=sum(view_set_1)/m1;
      cond_prob(k,12+1+(i-1)*2)=1-cond_prob(k,12+2+(i-1)*2);
      c=size(c);c=c(2);
      prob_act(k,i+6)=c/m1;
  end
end
% % 
% % %%% for frame 4
% cond_prob=zeros(cl_MAX,24); % p(x|a)
% %joint_prob=zeros(cl_MAX,24); % p(x,a)
% prob_act=zeros(cl_MAX,12); % p(a)
% state_prob=zeros(1,2); 
% 
% m=size(l4); m=m(1);
% m_1=size(l4_non);m_1=m_1(2);
% tot=size(f4_view)+size(f4_view_non);
% tot=tot(2);
% state_prob(2)= (sum(f4_view)+sum(f4_view_non)) / tot;  
% state_prob(1)= 1-state_prob(2);
% ind_set=0;view_set=0;
% ind_set_1=0;view_set_1=0;
% j=1;
% k=1;  
% for k=1:cl_MAX
%   ind_set= find(l4==k);
%   ind_set_1=find(l4_non==min_match_4(k));
%   m=size(ind_set);m=m(1);
%   m1=size(ind_set_1);m1=m1(1);
%   act_set=f4_act(ind_set);
%   view_set=f4_view(ind_set);
%   act_set_1=f4_act_non(ind_set_1);
%   view_set_1=f4_view_non(ind_set_1);
%   for i=1:6
%       c=find(act_set==i);
%       cond_prob(k,2+(i-1)*2)=sum(view_set)/m;
%       cond_prob(k,1+(i-1)*2)=1-cond_prob(k,2+(i-1)*2);
%       c=size(c);c=c(2);
%       prob_act(k,i)=c/m;
%   
%       c=find(act_set_1==i);
%       cond_prob(k,12+2+(i-1)*2)=sum(view_set_1)/m1;
%       cond_prob(k,12+1+(i-1)*2)=1-cond_prob(k,12+2+(i-1)*2);
%       c=size(c);c=c(2);
%       prob_act(k,i+6)=c/m1;
%   end
% end

% 



