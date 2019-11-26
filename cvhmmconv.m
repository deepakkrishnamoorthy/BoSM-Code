k=1;
i=1;
temp = [];
for j = 1:269
    temp = [temp;train1{k}];
    k=k+1;
    fprintf('\n Building submodel %d \n',i);
end
featMain1{i}=temp;

k=1;
i=1;
temp = [];
for j = 1:347
    temp = [temp;train2{k}];
    k=k+1;
    fprintf('\n Building through %d \n',k);
end
featMain2{i}=temp;
featMain = [featMain1 featMain2];
featCls = featMain;
clear featMain
M = 2; %mixtures
%M= 16;
%Q=[5 8 3 7 5 6 5];
Q = [5 5]
% Q = 6; %states 
numCls=2;
nex = 1;
featCls = cellfun(@(x) normr(x), featCls,'UniformOutput',false);

     
for i = 1:numCls
    O(i) = size(featCls{i},2);%dimension 
    T(i) = size(featCls{i},1);
    data = zeros(O(i),T(i),nex);
    data(:,:,nex) = featCls{i}'; 
    prior0{i} = normalise(rand(Q(i),1));
    transmat0{i} = mk_stochastic(rand(Q(i),Q(i))); 
    [mu0{i}, Sigma0{i}] = mixgauss_init(Q(i)*M,reshape(data, [O(i) T(i)*nex]) , 'diag'); %(mixture components,data,cov type)
    mu0{i} = reshape(mu0{i}, [O(i) Q(i) M]); 
    Sigma0{i} = reshape(Sigma0{i}, [O(i) O(i) Q(i) M]); 
    mixmat0{i} = mk_stochastic(rand(Q(i),M)); 
    [LL1{i}, prior1{i}, transmat1{i}, mu1{i}, Sigma1{i}, mixmat1{i}] = mhmm_em(data, prior0{i}, transmat0{i}, mu0{i}, Sigma0{i}, mixmat0{i}, 'max_iter', 5);
end
%%%Test features 
%testData = [MBH30_Col_NVCVTRAIN5 MBH30_Col_VCVTRAIN5];
testData = cellfun(@(x) normr(x), testData,'UniformOutput',false);
for i = 1:152
    test = testData{i};
    fprintf('\n testing video %d',i);
    for j = 1:2
        LogLikScore(i,j) = mhmm_logprob(test', prior1{j}, transmat1{j}, mu1{j}, Sigma1{j}, mixmat1{j});
        fprintf('\n Calculating logprob for video %d of fold %d',i,j);
    end   
end
[llkVal, llkLabel]=max(LogLikScore,[],2);
actuallabel1=[ones(55,1);2*ones(97,1)];
actuallabel=[zeros(55,1);ones(97,1)];
C=confusionmat(actuallabel1,llkLabel);
Accuracy=mean(actuallabel1==llkLabel)*100;
