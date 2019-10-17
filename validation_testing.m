clear all;
clc;
load('20_percent_data_prob.mat');

tot_cat=18;
cat_test_arr=[1 5 8 9 10 11 12 14 17];
k=size(cat_test_arr); k=k(2);

X=2;J=6;K=2;

i=0;x=0;
elem=0;
most_pref_act=zeros(X,tot_cat);
arr=zeros(1,J);
for i=1:k
    elem=cat_test_arr(i);
    elem
    if elem == 1
        for x=1:X
            arr=(1/state_prob(1,x))*(cond_prob(1,x:X:(J-1)*X+x).*prob_act(1,1:J));
            [~,most_pref_act(x,elem)] = max(arr);
        end
    else
        for x=1:X
            arr=(1/state_prob(1,x))*(cond_prob(elem-1,x+X*J:X:X*J+(J-1)*X+x).*prob_act(elem-1,J+1:J+J));
            [~,most_pref_act(x,elem)] = max(arr);
        end
        
    end
end


load('80_percent_data_prob.mat');
vldn_arr=zeros(8,2+2*X + 2*X);
count=0;count_1=0;m=0;

i=1;
for j=(i+1):18
    count=count+1;
    if  ( j==5 || j==8 || j==9 || j==10 || j==11 || j==12 || j==14 || j==17)
        count_1=count_1+1;
        
        [x,~]=milp_generalcost(J, K, X, cond_prob(count,:), prob_act(count,:));
        size(x)
        vldn_arr(count_1,1:2)=[i j];
        for m=1:X
            [~,vldn_arr(count_1,2+m)] = max(x(m:X:(J-1)*X+m));
            [~,vldn_arr(count_1,2+X+m)] = max(x(X*J+m:X:X*J+(J-1)*X+m));
        end
        vldn_arr(count_1,2+ 2*X + 1:2+ 2*X + 2*X) = abs(vldn_arr(count_1,2+1:2+2*X) - ...
            [most_pref_act(1,i) most_pref_act(2,i) most_pref_act(1,j) most_pref_act(2,j) ]);
        
    end
end