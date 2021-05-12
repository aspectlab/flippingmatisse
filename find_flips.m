% Inputs:
%   D - a 860x860 distance matrix
%   fnames - a list of filenames, stored as a cell array of character
%          vectors, where it is assumed that the list is ordered so that
%          RS/RT/VS/VT for each sample is contiguous (in other words,
%          every sample has its files ordered as a block in the order
%          _rr, _rt, _vr, _vt
%
% Outputs:
%   uniqueEds - list of all 50 editions
%   editions - edition string for each of the 860 image files
%   flip - shows, for each edition, which samples should be flipped.
%            0: no flip
%            1: [RS RT VS VT] --> [VS VT RS RT] (R/V flip)
%            2: [RS RT VS VT] --> [RT RS VT VS] (T/S flip)
%            3: [RS RT VS VT] --> [VT VS RT RS] (both R/V and T/S flip)
%   confidence - a confidence heuristic indicating the relative confidence
%          in suggested flips (if any) for each edition.  If no flips are
%          suggested, NaN appears as the confidence for that edition.
function [uniqueEds, editions, flip, confidence] = find_flips(D, fnames)

maxSPE = 10;  % largest number of samples per edition
N=size(fnames,1);  % total number of samples times four

% get edition of each sample
editions=cell(N,1);
for i=1:N
    editions{i}=fnames{i}(1:min(strfind(fnames{i},'_'))-1);
end
uniqueEds=unique(editions);
numEds=size(uniqueEds,1);

% permutations
P = [1 2 3 4   % RS RT VS VT
    3 4 1 2    % VS VT RS RT
    2 1 4 3    % RT RS VT VS
    4 3 2 1]'; % VT VS RT RS

% generate all permutations (saved in "p")
L=dec2base(0:4^(maxSPE-1)-1,4);  % string of all possible flips
L=fliplr([L char(ones(size(L,1),1)*'0')]);
p=zeros(4*maxSPE,4^(maxSPE-1));
for i=1:4^(maxSPE-1)
    for j=1:maxSPE
        p((j-1)*4+(1:4),i)=(j-1)*4+P(:,str2double(L(i,j))+1);
    end
end

% compute mask for summed distance calculation
m=zeros(4*maxSPE,1);
m(5:4:4*maxSPE)=ones(maxSPE-1,1);
mask=toeplitz(zeros(4*maxSPE,1),m');


% exhaustively check each permutation
flip=cell(numEds,1); % allocate space to store flipping results
confidence = NaN*ones(numEds,1);
for ed = 1:numEds  % loop through each edition
    idx=find(strcmp(editions,uniqueEds{ed})); % get indices of this edition
    Dedition=D(idx,idx);                      % extract distances for current edition
    num=size(idx,1)/4;                        % number of samples in this edition
    d=zeros(4^(num-1),1);                     % allocate space for summed distances for all permutations
    for i=1:4^(num-1)     % loop through each permutation
        d(i)=sum(sum(Dedition(p(1:4*num,i),p(1:4*num,i)).*mask(1:4*num,1:4*num)));
    end
    [min_d,idx2]=min(d);     % get permutation giving lowest sum distance
    flip{ed}=L(idx2,1:num);  % save string indicating best permutation
    if all(str2num(flip{ed}(2:end)')==1) && num>2  %#ok<ST2NM> % if we get something like '01111' make it '10000'
        flip{ed}=['1' '0'*ones(1,num-1)];
    end
    if all(str2num(flip{ed}(2:end)')==2) && num>2  %#ok<ST2NM> % if we get something like '02222' make it '20000'
        flip{ed}=['2' '0'*ones(1,num-1)];
    end
    
    if d(1) ~= min_d
        confidence(ed) = (d(1) - min_d)^2;    % somewhat arbitrary confidence heuristic
    end
    
end

% display nicely to screen
flipEdCtr = 0;
out = cell(sum(str2double(flip)~=0),1);   % create space for flipped editions
for ed = 1:numEds  % loop through each edition
    if str2double(flip{ed})~=0  % check if any flips in this edition
        flipEdCtr = flipEdCtr + 1;
        idx=find(strcmp(editions,uniqueEds{ed})); % get indices of this edition
        num=size(idx,1)/4;                        % number of samples in this edition
        for j=1:num
            sampleName = fnames{idx(j*4)}(1:end-3);
            switch flip{ed}(j)
                case '1'
                    out{flipEdCtr} = [out{flipEdCtr} ' + ' sampleName ': R/V' ];
                case '2'
                    out{flipEdCtr}=[out{flipEdCtr} ' + ' sampleName ': T/S' ];
                case '3'
                    out{flipEdCtr}=[out{flipEdCtr} ' + ' sampleName ': R/V + T/S' ];
            end
        end
        out{flipEdCtr}=out{flipEdCtr}(4:end); % remove leading +
        out{flipEdCtr}=[out{flipEdCtr} ' '*ones(1,35-length(out{flipEdCtr})) num2str(confidence(ed))];
    end
end

[~,order]=sort(confidence(~isnan(confidence)),'descend');
disp(' ')
disp('--------- Suggested flips w/confidence -----------------')
for i=1:flipEdCtr
    disp(out{order(i)})
end
disp('--------------------------------------------------------')
disp(' ')
