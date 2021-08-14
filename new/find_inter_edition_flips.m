load output_intra_flipped

% determine centroids and "most representative" sample for each edition
numEds=size(uniqueEds,1);
f_ctd=zeros(numEds*4,16);  % where we'll store centroid features
edition_rep = cell(numEds,1);
for ed=1:numEds
    idx=find(strcmp(editions,uniqueEds{ed})); % get indices of this edition
    num=size(idx,1)/4;
    D_temp = zeros(num,4);
    for j=1:4 % loop through the 4 orientations
        f_ctd((ed-1)*4+j,:)=mean(f(idx(j:4:end),:));  % compute centroid
        for k=1:num % loop through the samples in this edition
            D_temp(k,j)=sqrt(sum((f_ctd((ed-1)*4+j,:)-f(idx((k-1)*4+j),:)).^2));
        end
    end
    [~,idx2]=min(sum(D_temp,2));  % pick representative as one with smallest distance to centroid
    edition_rep{ed} = fnames{idx(idx2*4)}(1:end-3);      % store name of representative
end

% compute distance matrices for both the centroids and the representative samples
D_ctd = zeros(numEds*4);
for i=1:numEds*4
    for j=i+1:numEds*4
        D_ctd(i,j) = sqrt(sum((f_ctd(i,:)-f_ctd(j,:)).^2));
    end
end
D_ctd = D_ctd+D_ctd';  % exploit symmetry of distance matrix

% next: use D_ctd to find intra-edition
% flips, similar to how we did within an edition.  Now, though,
% we have 50 images so can't use brute search. We'll use a hierarchical
% approach: first partition 50 editions into 5 groups of 10, resolve flipping within
% each group of 10. Then combine and resolve flipping within final
% supergroup of 5.

NN=10;     % images per group
groups=5;  % number of groups (NN*groups = number of editions)

% deal with filenames
fnames_mock=cell(4*NN*groups,1);  % make up some fake filenames to trick find_flips
fnames2=cell(4*NN*groups,1);      % real filenames of representative images for make_mosaics
fnames_trimmed = cellfun(@(S) S(1:end-3), fnames, 'Uniform', 0);
suffix={'_rr','_rt','_vr','_vt'};
for g=1:groups
    for i=1:NN
        idx = find(strcmp(edition_rep{(g-1)*NN+i},fnames_trimmed),1);
        for j=1:4
            fnames_mock{(g-1)*NN*4+4*(i-1)+j}=['000' num2str(g) '_' num2str(i) suffix{j}];
            fnames2{(g-1)*NN*4+4*(i-1)+j}=fnames{idx+j-1};
        end
    end
end

% find the flips!
[uniqueEds2, editions2, flip2, ~] = find_flips(D_ctd, fnames_mock);    % centroid approach

% do flips within groups, according to prior results
allFlips2=cell2mat(flip2');
perm2=(1:size(fnames2,1))';
for i=1:size(allFlips2,2)
    switch allFlips2(i)
        case '1'  % R/V flip
            perm2(4*(i-1)+(1:4)) = perm2(4*(i-1)+[3 4 1 2]);
        case '2' % T/S flip
            perm2(4*(i-1)+(1:4)) = perm2(4*(i-1)+[2 1 4 3]);
        case '3' % both
            perm2(4*(i-1)+(1:4)) = perm2(4*(i-1)+[4 3 2 1]);
    end
end
f_ctd=f_ctd(perm2,:);
fnames_mock=fnames_mock(perm2);

% Computing second level of hierarchy
% Compute new centroids
f_ctd_of_ctds = zeros(groups*4, 16); % Where we store centroid of centroid features
for grp=1:groups
    idx=find(contains(fnames_mock, strcat('000', num2str(grp)))); % get indices of the group's images
    for j=1:4 % loop through the 4 orientations
        f_ctd_of_ctds((grp-1)*4+j,:) = mean(f_ctd(idx(j:4:end),:)); % compute centroid of centroids for each group
    end
end

% compute distance matrices
D_ctd_of_ctds = zeros(groups*4);
for i=1:groups*4
    for j=i+1:groups*4
        D_ctd_of_ctds(i,j) = sqrt(sum((f_ctd_of_ctds(i,:)-f_ctd_of_ctds(j,:)).^2));
    end
end
D_ctd_of_ctds = D_ctd_of_ctds+D_ctd_of_ctds';  % exploit symmetry of distance matrix

% deal with filenames
fnames_mock2=cell(4*groups,1);  % make up some fake filenames to trick find_flips
suffix={'_rr','_rt','_vr','_vt'};
for g=1:groups
    for j=1:4
        fnames_mock2{(g-1)*4+j}=['0001_' num2str(g) suffix{j}];
    end
end

[~, ~, flip3, ~] = find_flips(D_ctd_of_ctds, fnames_mock2);    % centroid of centroid approach

% apply flips to flips (i.e., flip "flip2" using "flip3" to create the
% cumulative edition flip)
flip22=zeros(groups,NN);
for i=1:groups
    for j=1:NN
        flip22(i,j)=bitxor(str2double(flip2{i}(j)),str2double(flip3{1}(i)));
    end
end
if sum(bitand(flip22,1),'all')>25   % if more than half are R/V flips, R/V everything (i.e., to minimize total # flips)
    flip22=bitxor(flip22,1);
end
if sum(bitand(flip22,2)/2,'all')>25 % if more than half are R/V flips, T/S everything (i.e., to minimize total # flips)
    flip22=bitxor(flip22,2);
end
flip22=flip22'; flip22=flip22(:);
[cell2mat(uniqueEds) repmat(' ',50,1) num2str(flip22)]
flip22s=cellstr(reshape(num2str(flip22),10,5)');

% create new permutation
permutation2=permutation;
for ed=1:numEds
    idx=find(strcmp(editions,uniqueEds{ed})); % get indices of this edition
    num=size(idx,1)/4;
    for j=1:num  % loop through each sample in the edition
        switch flip22(ed)
            case 1  % R/V flip
                permutation2(idx(4*(j-1)+(1:4))) = permutation2(idx(4*(j-1)+[3 4 1 2]));
            case 2 % T/S flip
                permutation2(idx(4*(j-1)+(1:4))) = permutation2(idx(4*(j-1)+[2 1 4 3]));
            case 3 % both
                permutation2(idx(4*(j-1)+(1:4))) = permutation2(idx(4*(j-1)+[4 3 2 1]));
        end
    end
end

% create new "flip" vector
flip_all=cell(size(flip));
for i=1:size(flip,1)
    flip_all{i}=num2str(bitxor(str2num(flip{i}'),flip22(i)))';
end

save permutation2 permutation2 flip_all

% make mosaics (for debugging)
%make_mosaics('data/matisse_color_small/', 'mosaics/', fnames2, uniqueEds2, editions2, flip22s)
