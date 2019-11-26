
%%%DATA PREPERATION
k = 1;
   m=1; 
for z=1 %model  only a single class
    for i = 1:60
    temp=[];
    if i ~= 76
        for j = 1:10  %10 segments per submodel
            fprintf('\n Entering video %d of sub model %d of class %d',j,i,z);
            temp=[temp;train{k}];
            k=k+1;
        end
    else
        for j = 1:8 %20 segments per submodel
            fprintf('\n Entering video %d of sub model %d of class %d',j,i,z);
            temp=[temp;train{k}];
            k=k+1;
        end
    end
    %featCls{z,i}=temp;
    featMain{m}=temp;
    m=m+1;
    end
end 


%featCls = cellfun(@(x) normr(x), featCls,'UniformOutput',false);
featMain = cellfun(@(x) normr(x), featMain,'UniformOutput',false);
%testing = cellfun(@(x) normr(x), testing,'UniformOutput',false);
training = cellfun(@(x) normr(x), training,'UniformOutput',false);
testing = cellfun(@(x) normr(x), testing,'UniformOutput',false);
%MBH_Col_Train =  cellfun(@(x) normr(x), MBH_Col_TrainNormal,'UniformOutput',false);
%do the above for forming representations
%clear MBH71_Col_Test MBH71_Col_Train;
featMain = cellfun(@(x) normr(x), featMain,'UniformOutput',false);

%%%Building the bag of models
M=2; %Mixtures
Q= [5*ones(1,76)]; %States

numCls=60;
nex = 1;

for i = 1:numCls
    fprintf('\n Building submodel %d \n',i);
    O(i) = size(featMain{i},2);%dimension 
    T(i) = size(featMain{i},1);
    data = zeros(O(i),T(i),nex);
    data(:,:,nex) = featMain{i}'; 
    prior0{i} = normalise(rand(Q(i),1));
    transmat0{i} = mk_stochastic(rand(Q(i),Q(i))); 
    [mu0{i}, Sigma0{i}] = mixgauss_init(Q(i)*M,reshape(data, [O(i) T(i)*nex]) , 'diag'); %(mixture components,data,cov type)
    mu0{i} = reshape(mu0{i}, [O(i) Q(i) M]); 
    Sigma0{i} = reshape(Sigma0{i}, [O(i) O(i) Q(i) M]); 
    mixmat0{i} = mk_stochastic(rand(Q(i),M)); 
    [LLC{i}, priorC{i}, transmatC{i}, muC{i}, SigmaC{i}, mixmatC{i}] = mhmm_em(data, prior0{i}, transmat0{i}, mu0{i}, Sigma0{i}, mixmat0{i}, 'max_iter', 5);
end

%%%Load training and testing data(extracted MBH features) and perform nomralization
training = cellfun(@(x) normr(x), training,'UniformOutput',false);
testing = cellfun(@(x) normr(x), testing,'UniformOutput',false);

%%%Forming training Embeddings
%train=[];
LogLikScoreT = zeros(600,60);
for i = 1:600
   train = training{i};
   fprintf('\n Training video %d',i);
    %trainData{i} = features{i};
   %train = MBH_Col_Train{i};
   for j=1:60
       fprintf('\n Calculating logprob for video %d of sub-model %d',i,j);
      LogLikScoreT(i,j) = mhmm_logprob(train', priorC{j}, transmatC{j}, muC{j}, SigmaC{j}, mixmatC{j});  
   end
end

%%%%Forming testing Embeddings
testing = cellfun(@(x) normr(x), testing,'UniformOutput',false);
LogLikScoreTT = zeros(152,76);


for i = 1:152
    %test = MBH_Col_Test{i};
    test = testing{i};
    for j = 1:76
        fprintf('\n Calculating logprob for video %d of fold %d',i,j);
        LogLikScoreTT(i,j) = mhmm_logprob(test', priorC{j}, transmatC{j}, muC{j}, SigmaC{j}, mixmatC{j});
    end   
end

%%%TRAINING THE SVM MODEL USING THE TRAINING AND TESTING EMBEDDINGS FORMED FROM THE ABOVE SEGMENT OF CODE
trainData = LogLikScoreT;
testData = LogLikScoreTT; 

%RUN FROM HERE FOR SVM
trainLabel = [0*ones(400,1);1*ones(400,1)];
testLabel = [0*ones(100,1); 1*ones(100,1)];
%ValidLabel = [0*ones(100,1); 1*ones(100,1)]; %Provide the labels for validation data

trainData = normr(trainData);
testData = normr(testData);
%validData = normr(validData) % uncomment and load desired validation data
    for i = 1:200
        Model = svmtrain(trainData,trainLabel,'kktviolationlevel',1,'method','SMO','kernel_function','rbf','rbf_sigma',(i));
        Group = svmclassify(Model,ValidData);
        ValAccuracy = mean(ValidLabel==Group)*100;
        SMONA5{i} = Accuracy;
        fprintf('\n Index1: %d',i); 
    end
    smo = max([SMONA5{:}]);

%%%Finally select the model with highest validation accuracy and provide testing data obtain the AUC and accuracy over the testing data.
