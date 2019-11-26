global DATAopts
DATAopts.videoPath = '%s';
clc;
clear;
blockSize = [6 6 6]; % block size is 6 by 6 pixels by 6 frames, but we will vary the number of frames
numBlocks = [3 3 2]; % 3 x3 spatial blocks and 2 temporal blocks
numOr = 8; % Quantization in 8 orientations
flowMethod = 'Horn-Schunck';
vidName = [pwd '/denis_bend.avi'];
vid = VideoReadNative(vidName);
%videoReadTime = toc;
%fprintf('Loaded video in %.2f seconds\n', videoReadTime);
hogDesc = cell(4, 1);
hogInfo = cell(4, 1);
idx = 1;
for frameSampleRate=[1 2 3 6]
    tic
    % Subsample framerate of video
    sampledVid = vid(:,:,1:frameSampleRate:end);
    % Get correct number of frames per block
    blockSize(3) = 6 / frameSampleRate;
    opticalFlow = Video2OpticalFlow(sampledVid, flowMethod);
end
    
    
    
