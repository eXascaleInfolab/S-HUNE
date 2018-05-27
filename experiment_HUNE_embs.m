%% HUNE: factorization based high-order unique node embedding
% load dataset
load('./blogcatalog.mat');

% Decay parameter (\alpha in the paper)
alpha_Katz = 0.9;
% Shifting parameter (\gamma in the paper)
K_shifted = 1;
% dimension of embeddings (d in the paper)
dim_emb = 128;

% learn embeddings
tic;
embs = hune(network, dim_emb, alpha_Katz, K_shifted);
toc;

% save embeddings
save('./embeddings_HUNE.mat','embs');






