numClusters=18;                      % number of clusters
clusterMethod = 'kmediods';          % either: 'hierarchical','kmeans','kmediods','spectral'
inPath='data/matisse_color_small/';  % path to small color images

% load the distances and features
load distances

% specify which model to use in clustering examples below
featureMethod = 'asil20';
D=eval(['D_' featureMethod]);
f=eval(['f_' featureMethod]);

% do clustering
switch clusterMethod
    case 'hierarchical'
        t1=cluster(linkage(squareform(D),'average'),'MaxClust',numClusters);
    case 'kmeans'
        t1=kmeans(f, numClusters,'replicates',25,'display','final');
    case 'spectral'
        t1=spectralcluster(f, numClusters);
    case 'kmediods'
        t1=kmedoids(f, numClusters, 'Distance', 'euclidean', 'replicates', 25);
    otherwise
        error('Invalid clustering method specified.')
end

% display clusters to screen
for i=1:numClusters
    disp(['Cluster ' num2str(i) ' has these textures in it: ']);
    disp(cellfun(@(S) S(1:end-3), fnames(find(t1==i)*4), 'Uniform', 0));
end

% create cluster mosaics
%clusterMosaic(t1, inPath, fnames)


% ************** This function generates the mosaic images *****************
function clusterMosaic(idx, inPath, fnames)

numClusters = size(unique(idx),1);

for C=1:numClusters
    figure(C)
    idx1=find(idx==C);
    
    % grab first texture (all 4 orientation) --> store in "img"
    temp=imread([inPath fnames{4*(idx1(1)-1)+1} '.png']);
    for i=2:4
        temp = cat(2, temp, imread([inPath fnames{4*(idx1(1)-1)+i} '.png']));
    end
    img=temp;
    
    % next, grab remaining textures
    clusterSize=size(idx1,1);
    if clusterSize > 20
        disp(['Cluster ' num2str(C) ' has ' num2str(clusterSize) ' textures, only showing 20 of them.'])
        clusterSize=20;
    end
    
    
    for k=2:clusterSize % start building mosaic
        temp=imread([inPath fnames{4*(idx1(k)-1)+1} '.png']);
        for i=2:4
            temp = cat(2, temp, imread([inPath fnames{4*(idx1(k)-1)+i} '.png']));
        end
        img=cat(1,img,temp);
    end
    
    imshow(img)
    title(['Cluster ' num2str(C)])
    
end
end
