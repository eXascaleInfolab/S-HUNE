function [embs,A] = hune(network, dim_emb, alpha_Katz, K_shited)
% HUNE: factorization-based graph embedding

% compute T_Katz
sp_radius = abs(eigs(network,1));
temp = inv(eye(size(network))-(alpha_Katz/sp_radius)*network);
T_Katz = zeros(size(network));
T_Katz = T_Katz + triu(temp) + triu(temp)'-eye(size(T_Katz)); % A = T_Katz;

% SPPMI transformation
temp_mat = T_Katz;
temp_mat = log(temp_mat*sum(sum(temp_mat))./(sum(temp_mat,2)*sum(temp_mat,1)))-log(K_shited);
temp_mat = temp_mat-diag(diag(temp_mat));
A = max(temp_mat,0);
% length(find(A))

% EVD
if(length(find(A))==0)
    embs = NaN;
    disp('K_shifted is set too high, SPPMI matrix are all zeros!');
else
    [V_emb,D_emb] = eigs(A,dim_emb,'la');
    
    temp = V_emb*sqrt(D_emb);
    embs = temp./(repmat(sqrt(sum(temp.^(2), 2)), 1, dim_emb));
    
    if sum(isnan(embs(:)))>0 || ~isreal(embs)
        embs = NaN;
        disp('A is singular or #postive eigenvalues are less than embedding dimension!');
    end
    
    
end






