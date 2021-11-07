clear ;
close all;
clc;
clean_speech_dir='.\timit\data\TRAIN\DR1\FCJF0\SI648.WAV';
noisy_speech_dir='.\timit_snr{snr}\timit\data\TRAIN\DR1\FCJF0\SI648.wav';
output_dir='.\Wiener Filter Enhanced Speech\enhanced_speech.wav';

[clean_speech,fs]=audioread(clean_speech_dir);      %read clean speech 
[noisy_speech,~]=audioread(noisy_speech_dir);      %read noisy speech
t=(0:length(clean_speech)-1)/fs;    
enhanced_speech=wiener_as(noisy_speech_dir,output_dir);      %apply Wiener Filtering,output the enhanced speech file 
t_=(0:length(enhanced_speech)-1)/fs;

figure();
subplot(321);
plot(t,clean_speech);ylim([-1.5,1.5]);title('Clean Speech');xlabel('Time/s');ylabel('Magnitude');
subplot(323);
plot(t,noisy_speech);ylim([-1.5,1.5]);title('Noisy Speech');xlabel('Time/s');ylabel('Magnitude');
subplot(325);
plot(t_,real(enhanced_speech));ylim([-1.5,1.5]);title('Wiener Filter Enhanced Speech');xlabel('Time/s');ylabel('Magnitude');
subplot(322);
spectrogram(clean_speech,256,128,256,16000,'yaxis');title('Spectrogram of Clean Speech');
subplot(324);
spectrogram(noisy_speech,256,128,256,16000,'yaxis');title('Spectrogram of Noisy Speech');
subplot(326);
spectrogram(enhanced_speech,256,128,256,16000,'yaxis');title('Spectrogram of Wiener Filter Enhanced Speech');colormap(jet);

