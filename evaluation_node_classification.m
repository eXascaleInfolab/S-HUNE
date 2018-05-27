% calssification evaluation using Deepwalk testing code: you need python
% with gensim, sklearn installed, double check python sys.path to make sure from matlab 
% or you have run the python command directly in the terminal
[status,cmdout] = system('python ./scoring.py ./blogcatalog.mat ./embeddings_HUNE.mat ./classification_res_HUNE.mat');

% get results
load('./classification_res_HUNE.mat');
F1 = squeeze(mean(res,1));
disp(F1);

