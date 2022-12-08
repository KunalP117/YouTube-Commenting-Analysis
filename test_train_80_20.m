clear all;clc;
load('full_data_4_frames.mat');
k=size(viewcount); k=k(2);

rng('shuffle');
p=randperm(k);


train_frac=0.8; test_frac=0.2;
thresh_ind=round(train_frac*k);

actions = actions(p(1:thresh_ind));
category = category(p(1:thresh_ind));
comments = comments(p(1:thresh_ind));
dislikes=dislikes(p(1:thresh_ind));
likes=likes(p(1:thresh_ind));
viewcount=viewcount(p(1:thresh_ind));
viewcount_disc=viewcount_disc(p(1:thresh_ind));

save('80_percent_data','actions','category','comments','likes','dislikes','viewcount','viewcount_disc');

load('full_data_4_frames.mat');

actions = actions(p(1+thresh_ind:k));
category = category(p(1+thresh_ind:k));
comments = comments(p(1+thresh_ind:k));
dislikes=dislikes(p(1+thresh_ind:k));
likes=likes(p(1+thresh_ind:k));
viewcount=viewcount(p(1+thresh_ind:k));
viewcount_disc=viewcount_disc(p(1+thresh_ind:k));

save('20_percent_data','actions','category','comments','likes','dislikes','viewcount','viewcount_disc');

