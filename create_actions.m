clear all;
clc;
load('n4_embed_data');
com_thresh=3;
sen_thresh=3;
s=size(lat_rep);
MAX=s(2);
actions=zeros(1,MAX);

i=0;
for i = 1:MAX
    if (comments(i)+1)>= exp(com_thresh)
        if abs(likes(i)-dislikes(i)) <= sen_thresh
            actions(i)=2;
        elseif likes(i)>= dislikes(i)
            actions(i)=3;
        elseif likes(i)<dislikes(i)
            actions(i)=1;
        end
    else
        if abs(likes(i)-dislikes(i)) <= sen_thresh
            actions(i)=5;
        elseif likes(i)>= dislikes(i)
            actions(i)=6;
        elseif likes(i)<dislikes(i)
            actions(i)=4;
        end
    end
end

desktop = com.mathworks.mde.desk.MLDesktop.getInstance;
desktop.restoreLayout('Default');
