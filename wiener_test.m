clear ;
close all;
clc;
d=dir('C:\Users\Yunyang Zeng\jupyter\mlsp\NMF_project\MLSP-FA2021-main\timit\data\TRAIN\DR1\**\*.wav');
speech_dir={d.name};
speech_folder={d.folder};

SNRdB = 0;
SNR = 10^(SNRdB/10);
rng('default');
snr_list=[];
for i=1:length(speech_dir)
%direction=append(speech_folder{i},speech_dir{i});
direction=strcat(speech_folder{i},'\',speech_dir{i});
if [direction(length(direction)-2) direction(length(direction)-1) direction(length(direction)-0)] ==['w','a','v']
    [clean_speech,fs]=audioread(direction);      %read clean speech 
    noise=normrnd(0,(var(clean_speech)/SNR)^0.5,length(clean_speech),1);
    noisy_speech=clean_speech+noise;
    snr_before=mean(clean_speech.^2)/mean(noise.^2);
    snr_before_db=10*log10(snr_before);
    [enhanced_speech]=wiener_as_v2(noisy_speech,fs);
    clean_speech=clean_speech(1:length(enhanced_speech));
    noisy_speech=noisy_speech(1:length(enhanced_speech));
    residual_noise=clean_speech-enhanced_speech';    
    snr_after=mean(clean_speech.^2)/mean(residual_noise.^2); 
    snr_after_db=10*log10(snr_after);
    snr_list=[snr_list [snr_before_db;snr_after_db]]; 
    
        
        
    
    
    
    end
end

average_snr_improvement=mean(snr_list(2,:)-snr_list(1,:));