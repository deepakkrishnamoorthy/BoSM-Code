clc;
clear;
blockSize = [6 6 6]; % block size is 6 by 6 pixels by 6 frames, but we will vary the number of frames
numBlocks = [3 3 2]; % 3 x3 spatial blocks and 2 temporal blocks
numOr = 8; % Quantization in 8 orientations
vidReader = VideoReader('denis_bend.avi');
opticFlow = opticalFlowHS;
flowMethod = 'Horn-Schunck'; % the optical flow choice
while hasFrame(vidReader)
     frameRGB = readFrame(vidReader);
     frameGray = rgb2gray(frameRGB);
     flow = estimateFlow(opticFlow,frameGray);
     nR = size(flow,1);
     nC = size(flow,2);
     nF = size(flow,3);
     extraPixelsR = mod(nR, blockSize(1));
     extraPixelsC = mod(nC, blockSize(2));
     extraFrames = mod(nF, blockSize(3));
     Vxbend1 = flow.Vx;
     Vybend1 = flow.Vy;
     OrBend1 = flow.Orientation;
     MagBend1 = flow.Magnitude;
     %HofBend2 = gradientHistogram(Vxbend1,Vybend1,260);
     imshow(frameRGB)
     hold on
     plot(flow,'DecimationFactor',[5 5],'ScaleFactor',25)
     hold off
end
