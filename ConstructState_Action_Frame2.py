import numpy as np
import scipy.io as sio
import os
import pandas as pd
from collections import Counter

os.chdir('/Users/whoiles/Desktop/DataPreperation')
data = sio.loadmat('deep_embedding_results')
data = pd.DataFrame({key: data[key][0].tolist() for key in ['category', 'viewcount', 'frame', 'dislikes', 'likes', 'comments']})

# data for frame1
dataf1 = data.loc[data['frame'] == 1]
print Counter(dataf1['category'])
# add state information for frame1
f1_viewcount_thr = 8
dataf1 = dataf1.assign(state = lambda row: (np.floor(np.log(row['viewcount']+1)) >= f1_viewcount_thr).astype(np.int8)+1)
Counter(dataf1['state'])
# add action information for frame1
def build_actions(dataf):
    f1_comment_thr = 3 # low/high comment threshold
    f1_likedislike_thr = 3 # otherwise neutral as not decisive
    likes = dataf['likes'].values
    dislikes = dataf['dislikes'].values
    comments = dataf['comments'].values
    actions = []
    for k in range(len(likes)):
        comment_high = np.floor(np.log(comments[k]+1)) >= f1_comment_thr
        sentiment_neut = (np.abs(likes[k] - dislikes[k]) <= f1_likedislike_thr)
        sentiment_hl = np.sign(likes[k] - dislikes[k])
        if comment_high and sentiment_neut:
            actions.append(2)
        elif comment_high and sentiment_hl == -1:
            actions.append(1)
        elif comment_high and sentiment_hl == 1:
            actions.append(3)
        elif not comment_high and sentiment_neut:
            actions.append(5)
        elif not comment_high and sentiment_hl == -1:
            actions.append(4)
        else:
            actions.append(6)
    return np.array(actions)

Counter(build_actions(dataf1))

dataf1['actions'] = build_actions(dataf1)

# add decisionproblem
category_cnt = Counter(dataf1['category'])
most_common_cat = category_cnt.most_common(n=1)[0][0]
dataf1 = dataf1.assign(decision = lambda row: (row['category'].values == most_common_cat).astype(np.int8)+1)
Counter(dataf1['decision'])

# compute conditional probability p(x|a) and p(a) for each decision problem
J = 6 # number of actions
K = 2 # number of decision problems
X = 2 # number of states

prob_act = np.zeros(J*K)
cond_prob = np.zeros(X*J*K)
for j in [1,2,3,4,5,6]:
    for i in [1,2]:
        for k in [1,2]:
            Nk = np.sum(dataf1['decision'] == k) # total number of samples in decision problem
            joint_sum = Counter(dataf1.loc[(dataf1['decision'] == k) & (dataf1['state'] == i)]['actions'])
            action_sum = Counter(dataf1.loc[(dataf1['decision'] == k)]['actions'])
            prob_act[j+(k-1)*J-1] = action_sum[j]/float(Nk)
            cond_prob[i+(j-1)*X+(k-1)*X*J-1] = joint_sum[j]/float(action_sum[j])

frame2 = {'prob_act': prob_act, 'cond_prob': cond_prob}
sio.savemat('frame2_probability.mat', frame2)










