
% pause(1);
% Setup a global variable which contains the path of video files
global DATAopts
DATAopts.videoPath = '%s';

% HOG/HOF settings
blockSize = [12 12 1]; % block size is 12 by 12 pixels by 1 frames, but we will vary the number of frames
numBlocks = [3 3 2]; % 3 x3 spatial blocks and 2 temporal blocks
numOr = 8; % Quantization in 8 orientations
flowMethod = 'Horn-Schunck'; % For HOF only

%fprintf('\n Entering fold : %d ',s);
trainFiles =  dir(strcat('D:\HAR DATASETS av vio etc\HAR Dataset\Violent Flow Datasets\5\Violence\*.avi')); %provide directory of training videos

trainfiles = length(trainFiles);
%testfiles = length(testFiles);

s=1;
for i=1:trainfiles
fprintf('\n Entering train video : %d',i);
% Load the video
vidName = strcat('D:\HAR DATASETS av vio etc\HAR Dataset\Violent Flow Datasets\5\Violence\',trainFiles(i).name);
vid = VideoReadNative(vidName);
dim = size(vid);
frames = dim(3);
b=1;
e=25;
frameSampleRate=1;
while e<=frames 
    trainSegments{s} = vid(:,:,b:e);
    b=b+25;
    e=e+25;
    % Subsample framerate of video
    sampledVid = trainSegments{s}(:,:,:);
    
    % Get correct number of frames per block
    blockSize(3) = 6 / frameSampleRate;
    
    % Get HOG descriptors
    [MBH30_Row_Train{1,s}, MBH30_Col_VCVTRAIN5{1,s}] =  Video2DenseMBHVolumes(sampledVid, blockSize, numBlocks, numOr, flowMethod);
    s = s+1;
    
    % Get MBH descriptors
   %    [MBHRowDesc_Train{1,s}, MBHColDesc_Train{1,s}, mbhInfo_Train{1,s}] = ...
   %    Video2DenseMBHVolumes(sampledVid, blockSize, numBlocks, numOr, flowMethod);
   %    s=s+1;
end
end


%save MBHColDesc_Train{1,s} varaible to train the model based on the number of video segments formed above


