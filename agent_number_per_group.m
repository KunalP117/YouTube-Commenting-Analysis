load('n4_embed_data.mat');
load('full_data_4_frames.mat');
cat=unique(category);
count_arr_1=zeros(3,18); %1st row avg. comments, 2nd row avg. likes+dislikes
count_arr_2=zeros(3,18);
count_arr_3=zeros(1,18);
count_arr_4=zeros(2,18);
x=0;y=0;z=0;x_arr=0;y_arr=0; j=0;
for i=1:18
    count_arr_1(1,i) = sum(comments(category==cat(i)));
    count_arr_1(2,i) = sum(likes(category==cat(i))+dislikes(category==cat(i)));
    count_arr_1(1,i)=count_arr_1(1,i)/sum((category==cat(i)));
    count_arr_1(2,i)=count_arr_1(2,i)/sum((category==cat(i)));
    count_arr_1(3,i)=max([count_arr_1(1,i) count_arr_2(2,i)]);
 
    count_arr_2(1,i) = max(comments(category==cat(i)));
    count_arr_2(2,i) = max(likes(category==cat(i))+dislikes(category==cat(i)));
    %count_arr_2(1,i)=count_arr_2(1,i)/sum((category==cat(i)));
    %count_arr_2(2,i)=count_arr_2(2,i)/sum((category==cat(i)));
    count_arr_2(3,i)=max([count_arr_2(1,i) count_arr_2(2,i)]);
    
    x=comments(category==cat(i));
    y=likes(category==cat(i))+dislikes(category==cat(i));
    z=horzcat(x(x>=y),y(y>x));
    count_arr_3(i)=sum(z)/sum((category==cat(i)));
    
    x=comments(category==cat(i));x_arr=viewcount_disc(category==cat(i));
    x=x(x_arr==1);
    y=likes(category==cat(i)) + dislikes(category==cat(i));y_arr=viewcount_disc(category==cat(i));
    y=y(y_arr==1);
    k=size(x);k=k(2);
    for j=1:k
        z(j)=max([x(j) y(j)]);
    end
    %z=horzcat(x(x>=y),y(y>x));
    %k=size(x);k=k(2);
    count_arr_4(1,i)=sum(z)/k;
    count_arr_4(3,i)=max(z);
    
    x=comments(category==cat(i));x_arr=viewcount_disc(category==cat(i));
    x=x(x_arr==0);
    y=likes(category==cat(i)) + dislikes(category==cat(i));y_arr=viewcount_disc(category==cat(i));
    y=y(y_arr==0);
    k=size(x);k=k(2);
    for j=1:k
        z(j)=max([x(j) y(j)]);
    end
    count_arr_4(2,i)=sum(z)/k;
    count_arr_4(4,i)=max(z);
    
end
avg_max_users=sum(count_arr_2(3,:))/18;