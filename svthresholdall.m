trainData = LogLikScoreT;
testData = LogLikScoreTT; 
trainData = logLikT;
testData = logLikTT;
 
trainData=normr(trainData);
testData=normr(testData);
trainData=train;
 
%RUN FROM HERE FOR SVM
trsam=306;
ttsam=296;
trainLabel=[ones(trsam,1)*1;];
testLabel=test_label;
testLabel = testLabel(1,1:ttsam);
for i = 1:ttsam
    if(testLabel(1,i) == 0)
        testLabel(1,i)= 1;
    else
        testLabel(1,i)=-1;
    end
end
testLabel=testLabel';
 
model = svmtrain(trainLabel, trainData, '-s 2 -t 0 -c 1.5 -g 1.75  -d 2');
[segment_anamoly,acc,cmat] = svmpredict(testLabel, testData, model);
cmat = confusionmat(testLabel, segment_anamoly) 
 
 
[X,Y,T,AUC]=perfcurve(testLabel,segment_anamoly,'-1')
lgd = legend(['AUC = ', num2str(AUC, '%4.2f'),],...
              'FontSize',24,'location','east')
plot(X,Y,':b','LineWidth',2.5)
set(gca,'fontsize',22)
xlabel('False positive rate','FontSize', 24); 
ylabel('True positive rate','FontSize', 24);
title('ROC Curve of Test Data')
 
%min
for row = 1:trsam
    avg=0;
    avg = min(trainData(row,:));
    average(row,1)=avg;
end
threshold = min(average)
for trow = 1:ttsam
    segmin = min(testData(trow,:));
    if(segmin<threshold)
        segment_anamoly(trow,1)=1;
    else
        segment_anamoly(trow,1)=0;
    end
end
 
ccount=0;
anc=0;
nc=0;
for val = 1:ttsam
    if(segment_anamoly(val,1)==test_label(1,val))
        ccount=ccount+1; 
    end
end
accuracy = ccount/ttsam * 100
cmat = confusionmat(segment_anamoly,test_label(1,1:ttsam)')
 
[X,Y,T,AUC]=perfcurve(test_label,segment_anamoly,'1')
lgd = legend(['AUC = ', num2str(AUC, '%4.2f'),],...
              'FontSize',24,'location','east')
          plot(X,Y,':b','LineWidth',2.5)
 
set(gca,'fontsize',22)
xlabel('False positive rate','FontSize', 24); 
ylabel('True positive rate','FontSize', 24);
title('ROC Curve of Test Data')
 
 
%average 
for row = 1:trsam
    avg=0;
    avg=mean(trainData(row,:));
    average(row,1)=avg;
end
threshold=mean(average)
for trow = 1:ttsam
    segthr = mean(testData(trow,:));
    segmenthresh(trow,1) = segthr;   %scores
    if(segthr<=threshold)
        segment_anamoly(trow,1)=1;
    else
        segment_anamoly(trow,1)=0;
    end
end
 for i = 1:ttsam
    if(testLabel(1,i) == -1)
        testLabel(1,i)=1;
    else
        testLabel(1,i)=0;
    end
end
testLabel=testLabel';

ccount=0;
anc=0;
nc=0;
for val = 1:ttsam
    if(segment_anamoly(val,1)==testLabel(1,val))
        ccount=ccount+1; 
    end
end
accuracy = ccount/ttsam * 100
cmat = confusionmat(segment_anamoly,test_label(1,1:ttsam)')
 
[X,Y,T,AUC]=perfcurve(testLabel(1,1:ttsam)',segment_anamoly,'1')
lgd = legend(['AUC = ', num2str(AUC, '%4.2f'),],...
              'FontSize',24,'location','east')
 
set(gca,'fontsize',22)
xlabel('False positive rate','FontSize', 24); 
ylabel('True positive rate','FontSize', 24);
title('ROC Curve of Test Data')

load('H:\Submissions\Score Vector\Experiments Results\CV to be tested python\ACTUAL F3\train.mat')
load('H:\Submissions\Score Vector\Experiments Results\CV to be tested python\ACTUAL F3\test.mat')

