load output_asil20 % this data file is created by passing the Matisse dataset through the Asilomar model (To recreate: run flipping.py with MODEL_FILENAME = 'asil20_weights', then rename output.mat as 'output_asil20.mat')
load permutation2  % load permutations which were created using the fm model

% do all flips 
D=D(permutation2,permutation2);
f=f(permutation2,:);
fnames=fnames(permutation2);

% given image distances, compute texture distances (1 texture = 4 images)
num=size(D,1);
D2=zeros(num/4);
for i=1:num/4
    for j=1:num/4
        D2(i,j)=sqrt(sum(D(sub2ind([num num],4*(i-1)+(1:4),4*(j-1)+(1:4))).^2));
    end
end

% create augmented feature vectors of size 16x4 = 64
f2=zeros(215,64);
for i=1:215
    f2(i,:)=[f(4*(i-1)+1,:) f(4*(i-1)+2,:) f(4*(i-1)+3,:) f(4*(i-1)+4,:)];
end

D_asil20 = D2;
f_asil20 = f2;
save distances D_asil20 f_asil20 fnames
