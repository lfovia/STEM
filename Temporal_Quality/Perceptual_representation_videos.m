 clc;
 clear all;
 close all;
%%%%%%%%%%%%%
addpath('/home/parimala/Desktop/LIVE_VQC/Video')
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
real_path = dir(fullfile('/home/parimala/Desktop/LIVE_VQC/Video/*.mp4'));

real=size(real_path)

for ii=1:1:real
    
    DistVideoName1 = real_path(ii).name;
    RefVideoName = VideoReader(DistVideoName1);
    length = RefVideoName.NumberOfFrames
RefVideoName = VideoReader(DistVideoName1);
Frame_NIQE = [];
LGN_features_level6 =[];
for rr = 1:1:length

frameRGB = readFrame(RefVideoName);
    frameGray = double(rgb2gray(frameRGB));
    [y l] = frame_LGN_features(frameGray);
    LGN_features_level6(rr,:) = y{6}(:);
    
end  
save(strcat('./LGN6_VQC_new/',DistVideoName1,'.mat'),'LGN_features_level6')
end
