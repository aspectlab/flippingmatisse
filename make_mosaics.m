function make_mosaics(inPath, outPath, fnames, uniqueEds, editions, flip)
numEds=size(uniqueEds,1);
% permutations
P = [1 2 3 4    % RS RT VS VT
    3 4 1 2     % VS VT RS RT
    2 1 4 3     % RT RS VT VS
    4 3 2 1]';  % VT VS RT RS

for ed=1:numEds
    
    if ~all(str2num(flip{ed}')==0)  %#ok<ST2NM> % only generate mosaic if there's a flip
        idx=find(strcmp(editions,uniqueEds{ed}));  % original order
        num=size(flip{ed},2);
        
        % generate flipped order
        idx2=zeros(size(idx));
        for j=1:num
            idx2(4*(j-1)+(1:4))=4*(j-1)+P(:,str2double(flip{ed}(j))+1);
        end
        idx2=idx(idx2);
        
        % generate orig image tiling
        ctr=0;
        for j=1:num
            ctr=ctr+1;
            temp=imread([inPath fnames{idx(ctr)} '.png']);
            for i=2:4
                ctr=ctr+1;
                temp = cat(2, temp, imread([inPath fnames{idx(ctr)} '.png']));
            end
            if j==1
                f=temp;
            else
                f=cat(1,f,temp);
            end
        end
        imwrite(f,[outPath uniqueEds{ed} '_orig.png'])
        
        % generate permuted image tiling
        ctr=0;
        for j=1:num
            ctr=ctr+1;
            temp=imread([inPath fnames{idx2(ctr)} '.png']);
            for i=2:4
                ctr=ctr+1;
                temp = cat(2, temp, imread([inPath fnames{idx2(ctr)} '.png']));
            end
            if j==1
                f=temp;
            else
                f=cat(1,f,temp);
            end
        end
        imwrite(f,[outPath uniqueEds{ed} '_flip.png'])
    end
end
