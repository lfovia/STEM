clc;
clear all;
close all;

%%%%%%%%%%%%%
%addpath('./All_videos/');
FULL_PATH = dir(fullfile('./LGN6_VQC/*.mat'));
load data.mat
FULL_PATH_2_1 = dir(fullfile('./NIQE_VQC_Scores/*.mat'));
width_video = [];
Mos_scores = [];
features_norm21_rmse = [];
features_norm1 =[];
features_norm2 =[];
features_norm2_mean =[];
features_norm3 =[];
features_norm4_mssimvar =[];
features_norm4_mssimmean =[];
features_niqe_hysteresis = [];
features_rmse_recency = [];
features_rmse_hysteresis = [];
features_norm1 =[];
features_norm2 =[];
features_norm2_mean = [];
features_norm3 =[];
features_norm4_mssimvar =[];
features_norm4_mssimmean =[];
features_norm5 =[];
features_norm6 =[];
features_norm7 =[];
features_norm8 =[];
features_norm9 =[];
features_norm10 =[];
features_norm11 =[];
features_norm12 =[];
features_norm13 =[];
features_norm14 =[];
features_norm15 =[];
features_norm16=[];
features_norm17 = [];
features_norm18=[];
features_norm19 = [];
features_norm20 =[];
features_norm21 =[];
features_norm22 =[];
features_norm23 =[];
features_norm24 =[];
%features_norm17=[];
%features_norm1 =[];
Mos_Scores = [];
feature_RMSE = [];
feature_RMSE2 = [];
feature_RMSE_distance = [];
feature_RMSE_distance2 = [];
feature_RMSE_mahal = [];
feature_RMSE_cosine = [];
feature_RMSE_correlation = [];
for k=1:1:585
    name_folder = strcat(video_list{k},'.mat');
    name3 = strcat('./LGN6_VQC_new/',name_folder);
    
    load(name3);
    name4 = strcat('./NIQE_VQC_Scores/',name_folder);
    
    load(name4);

    
    features_norm3 =[features_norm3;mean(Frame_NIQE)];
LGN_features_level6(isnan(LGN_features_level6)|isinf(LGN_features_level6))=0;
 
    features_norm22 =[];
    niqe_scores = (Frame_NIQE)';
TrainData = LGN_features_level6';
    TrainMean = mean(TrainData,2); % Total mean of the training set
    n = size(LGN_features_level6,2);
    TotalTrainSamples = size(LGN_features_level6,1);
Gt=zeros([ n n]);
for i=1:TotalTrainSamples
    Temp = TrainData(:,i)- TrainMean;
    Gt = Gt + Temp'*Temp;
end
Gt=Gt/TotalTrainSamples; 

% Applying eigen-decompostion to Gt and returning transformation matrix
% 
%---------------------------------------------------------------------------------
[EigVect1,EigVal1]=eig_decomp(Gt);
EigVect=EigVect1(:,1:10); 

% Deriving training feature matrices
%----------------------------------------------------------------------------------

for i=1:TotalTrainSamples
    Ytrain(:,i)=(TrainData(:,i)'*EigVect)';
end
    
    ll = size(niqe_scores,1);
    xx =round(ll/4)
 for jj=1:1:xx
        nn = jj:-1:1;
        weights = exp(-0.1*nn);
      weights2 = weights(1:jj)./sum(weights);
      ww = weights2./(sum(weights2));
      features_norm22(jj) = sum(niqe_scores(1:jj)'.*ww);
        
    end
    C_mse = [];
    for oo =1:1:size(LGN_features_level6,1)-8
        xx2 = LGN_features_level6(oo:oo+7,:);
        yy = LGN_features_level6(oo+8,:);
        model = fitlm(xx2',yy');
        C_mse = [C_mse;model.RMSE];

        
    end
    feature_RMSE = [feature_RMSE;mean(C_mse)];

C_mse2 = [];
    Ytrain2 =Ytrain';
    for oo =1:1:size(LGN_features_level6,1)-4
        xx22 = Ytrain2(oo:oo+3,:);
        yy22 = Ytrain2(oo+4,:);
        model = fitlm(xx22',yy22');
        C_mse2 = [C_mse2;model.RMSE];
		
        
    end
    feature_RMSE2 = [feature_RMSE2;mean(C_mse2)];
      
      nn = xx:-1:1;
      nn_2 = 1:1:ll;
      weights = exp(-0.1*nn);
      weights2 = weights(1:xx)./sum(weights);
      ww = weights2./(sum(weights2));
      for kk=xx:1:ll
          features_norm22(kk) = sum(niqe_scores(kk-xx+1:kk)'.*ww);
      end
      
      features_norm21 = [features_norm21;mean(features_norm22)]; 
    
    Mos_Scores = [Mos_Scores;mos(k)];
      
end
Mos_scores = Mos_Scores;
hh = [features_norm21,log(feature_RMSE2)];                    
correlation = [correlation;calculatepearsoncorr(mean(hh,2),Mos_scores)]

