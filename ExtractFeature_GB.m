%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Maral Kasiri- Sepehr Jalali
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file extract the features from the feature vector 
% for Gradient Boosting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
clear all;
clc;
load('Data_Input.mat')
Feature_train=ExtractFeature(Data_train,Label_train);
Feature_test= ExtractFeature(Data_test,Label_test);
save('Feature_GB.mat')


function featureVector= ExtractFeature(Data, Label)

[m,n,k]= size(Data);
feature_set= zeros(m,47);
for j=1:2
    for i=1:m
    [c,l]= wavedec(Data(i,:,j),5,'db4');
    approx=appcoef(c,l,'db4');
    [cd1,cd2,cd3,cd4, cd5]= detcoef(c,l,[1 2 3 4 5]);
    MAbsVal= [mean(abs(cd1)), mean(abs(cd2)),mean(abs(cd2)),mean(abs(cd2)),mean(abs(cd2)),mean(abs(approx))];
    AvPower= [bandpower(cd1),bandpower(cd2),bandpower(cd3),bandpower(cd4),bandpower(cd5),bandpower(approx)];
    Sdev= [std(cd1),std(cd2), std(cd3),std(cd4),std(cd5),std(approx)];
    Ratio= [abs(mean(cd2))/ abs(mean(cd1)), abs(mean(cd3))/ abs(mean(cd2)), abs(mean(cd4))/ abs(mean(cd3)), abs(mean(cd5))/ abs(mean(cd4)), abs(mean(approx))/ abs(mean(cd5))];
    
    if j==1
        feature_set(i,1:24)= [Label(i,1) MAbsVal AvPower Sdev Ratio];
    else
        feature_set(i,25:end)= [MAbsVal AvPower Sdev Ratio];
    end
    
    end
end
   
featureVector=feature_set;
end
