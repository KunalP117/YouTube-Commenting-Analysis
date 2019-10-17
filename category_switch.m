clc;
clear all;

load('full_data_4_frames.mat');
%load('20_percent_data.mat');
C=0;
ia=0;
ic=0;
[C,ia,ic]=unique(category);
cat_num=size(C); cat_num=cat_num(2);

prob_act=zeros(cat_num*(cat_num-1)/2,12);
cond_prob=zeros(cat_num*(cat_num-1)/2,24);
joint_prob=zeros(cat_num*(cat_num-1)/2,24);
state_prob=zeros(cat_num*(cat_num-1)/2,2);

count=0;
act=0; %action index
state=1; %state index
k=0; %decision problem index
act=1 ; %(action index)

dummy=0;
dummy_1=0;
num=0;
for i=1:(cat_num-1)
    for j=(i+1):cat_num
        count=count+1;
        for act=1:6
            dummy=actions(category==C(i)); 
            num=size(dummy);num=num(2);
            dummy=dummy(dummy==act);
            dummy=size(dummy); dummy=dummy(2);
            prob_act(count,act)= dummy/num;

            dummy=actions(category==C(j)); 
            num=size(dummy);num=num(2);
            dummy=dummy(dummy==act);
            dummy=size(dummy); dummy=dummy(2);
            prob_act(count,6+act)= dummy/num;

            %for state=1:2
            dummy=actions(category==C(i));
            dummy_1=viewcount_disc(category==C(i));
            dummy_1=dummy_1(dummy==act);
            num=size(dummy_1);num=num(2);
            dummy=sum(dummy_1);
            if num ~=0
                cond_prob(count,act*2)=dummy/num;
            end
            joint_prob(count,act*2)= cond_prob(count,act*2)*prob_act(count,act);
            cond_prob(count,act*2-1)=1 - cond_prob(count,act*2);
            joint_prob(count,act*2-1)= cond_prob(count,act*2-1)*prob_act(count,act);
            
            
            dummy=actions(category==C(j));
            dummy_1=viewcount_disc(category==C(j));
            dummy_1=dummy_1(dummy==act);
            num=size(dummy_1);num=num(2);
            dummy=sum(dummy_1);
            if num ~=0
                cond_prob(count,12 + act*2)=dummy/num;
            end
            joint_prob(count,12 + act*2)= cond_prob(count,12 + act*2)*prob_act(count,6 + act);
            cond_prob(count,12 + act*2-1)=1 - cond_prob(count,12 + act*2) ;
            joint_prob(count,12 + act*2 - 1)= cond_prob(count,12 + act*2 - 1)*prob_act(count,6 + act);
%             
%             
               
            %end
        end
    end
end
count=0;
for i=1:cat_num-1
    for j=i+1:cat_num
        count=count+1;
        dummy=horzcat(viewcount_disc(category==C(i)),viewcount_disc(category == C(j)));
        k=size(dummy);k=k(2);
        state_prob(count,2)=sum(dummy)/k;
        state_prob(count,1)=1-state_prob(count,2);
    end
end
