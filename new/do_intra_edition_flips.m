% This script does the intra-edition flips, and re-permutes the distance matrix,
% feature vectors, filenames, etc.  It first makes some adjustments
% to algorithm suggestions based on visual inspection, primarily
% deciding *not* to flip obvious texture outliers that clearly
% don't fit with the edition, and it ignores some lower confidence
% suggestions.  That is, we err on the side of 'not' flipping when
% it's not clear (based on a non-expert visual inspection) what to do.
%
% Algorithm suggests (in decreasing order of confidence) --
%  *2603_essai: T/S                    23.9451
%  *1249_essai_10: R/V                 21.2374
%  *1313_epreuve_10: R/V               7.2488
%  *1324_premier_etat: R/V             5.69
%  *1403_32: R/V + 1403_33: R/V        3.4984
%  *1327_29: R/V                       3.2276
%   1301_essai_japon: R/V              2.8947
%  *2607_8: R/V + 2607_18: R/V         2.8446
%   1256_essai_japon: R/V              2.44
%  *2600_essai: R/V                    2.2362
%   1300_4: R/V                        1.1208
%  *2592_essai_b: R/V                  1.0508
%   2589_11: R/V + 2589_18: R/V        0.51924
%  *1367_3: R/V                        0.43092
%   2598_essai_b: R/V                  0.16849
%   1348_21: R/V                       0.16247
%   2594_3: R/V                        0.015796
%   2601_16: R/V                       0.0072236
%
% This script instead implements these 13 flips --
%  *1249_essai_10: R/V
%  *1313_epreuve_10: R/V
%  *1324_premier_etat: R/V
%  *1327_29: R/V
%   1362_17: R/V
%  *1367_3: R/V
%  *1403_32: R/V + 1403_33: R/V
%  *2592_essai_b: R/V
%  *2600_essai: R/V
%  *2603_essai: T/S
%  *2607_8: R/V + 2607_18: R/V
%
% Note that starred (*) entries are in common to both the algorithm and the manual adjustment.

load ../output
addpath('..')
[uniqueEds, editions, flip] = find_flips(D, fnames);
% make changes, relative to algorithm suggestion
flip{3}= '0000';      % skip 1256_essai_japon
flip{11}='00000';     % skip 1300_4
flip{12}='0000000';   % skip 1301_essai_japon
flip{18}='000';       % skip 1348_21
flip{32}='0000';      % skip 2589_11 + 2589_18
flip{35}='000';       % skip 2594_3
flip{39}='0000';      % skip 2598_essai_b
flip{42}='0000';      % skip 2601_16
flip{23}='00100';     % add in new flip: 1362_17

% given suggested flips, compute permutation
permutation=(1:860)';
for i=1:size(uniqueEds,1)
    if ~all(str2num(flip{i}')==0) %#ok<ST2NM>
        ed=uniqueEds{i};
        idx=find(strcmp(editions,ed));
        for j=1:size(flip{i},2)
            switch flip{i}(j)
                case '1'
                    permutation(idx(4*(j-1)+(1:4)))=permutation(idx(4*(j-1)+[3 4 1 2]));
                case '2'
                    permutation(idx(4*(j-1)+(1:4)))=permutation(idx(4*(j-1)+[2 1 4 3]));
                case '3'
                    permutation(idx(4*(j-1)+(1:4)))=permutation(idx(4*(j-1)+[4 3 2 1]));
            end
        end
    end
end

% do intra-edition flips
D=D(permutation,permutation);
f=f(permutation,:);
fnames=fnames(permutation);
save output_intra_flipped D editions f flip fnames permutation uniqueEds
