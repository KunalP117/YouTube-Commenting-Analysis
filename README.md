# YouTube_project
This repository contains data as well as code for analyzing rationally inattentive commenting behavior in YouTube. 
The formal theory and model is contained in the paper found at this link: https://arxiv.org/abs/1910.11703 .
The raw YouTube data can be found at the public Google Drive folder: https://drive.google.com/drive/folders/1ByvDYQzZR6hHfWle5EXhBkoa4YpomNvF?usp=sharing. The data files need to be unwrapped through the pickle module in python using the youtube style file included in this repository. 

The repository contains the following:
1. Raw YouTube data consisting of viewcount, comment count, video ratings (likes and dislikes), thumbnail, description of each individual video.
2. Code for pre-processing raw data to generate probability mass functions of state, action, (state,action) and conditional probability mass functions of action given state. (State -> Viewcount, Action -> Comment count, video rating)
3. Code for Decision test for utility maximization under general cost, renyi mutual information cost, shannon mutual information cost. 
4. Code for Robustness test to check deviation from optimal behavior.
