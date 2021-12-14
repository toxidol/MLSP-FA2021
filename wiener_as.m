function [enhanced_speech] = wiener_as(noisy_speech,fs)
% column vector noisy_speech

% set parameter values
mu= 0.98; % smoothing factor in noise spectrum update
a_dd= 0.98; % smoothing factor in priori update
eta= 0.15; % VAD threshold
frame_dur= 20; % frame duration (20ms Hamming Window)
frame_len= frame_dur* fs/ 1000; % L is frame length (160 for 8k sampling rate)
ham_window= hamming( frame_len); % hamming window
U= ( ham_window'* ham_window)/ frame_len; % normalization factor

%assume first 120 ms is noise only
len_first_120ms= fs/ 1000* 120;
first_120ms= noisy_speech( 1: len_first_120ms);

%use Welch's method to estimate PSD function
n_sub_frames= floor( len_first_120ms/ (frame_len/ 2))- 1;  % 50% overlap
noise_psd= zeros( frame_len, 1); 
n_start= 1; 
for i= 1: n_sub_frames
    noise= first_120ms( n_start: n_start+ frame_len- 1);
    noise= noise.* ham_window;
    noise_fft= fft( noise, frame_len); %noise FFT
    noise_psd= noise_psd+ ( abs( noise_fft).^ 2)/ (frame_len* U);
    n_start= n_start+ frame_len/ 2; 
end
noise_psd= noise_psd/ n_sub_frames;
% number of noisy speech frames 
len1= frame_len/ 2; % with 50% overlap
nframes= floor( length( noisy_speech)/ len1)- 1; 
n_start= 1; 
for i= 1: nframes
    noisy= noisy_speech( n_start: n_start+ frame_len- 1);
    noisy= noisy.* ham_window;
    noisy_fft= fft( noisy, frame_len);
    noisy_ps= ( abs( noisy_fft).^ 2)/ (frame_len* U);
    
    %======  VAD ==========
    if (i== 1) % initialize posteri
        post= noisy_ps./ noise_psd; %postriori SNR
        posteri_prime= post- 1; 
        posteri_prime( find( posteri_prime< 0))= 0;
        priori= a_dd+ (1-a_dd)* posteri_prime; %a priori SNR
    else
        post= noisy_ps./ noise_psd; %postriori SNR
        posteri_prime= post- 1;
        posteri_prime( find( posteri_prime< 0))= 0;
        priori= a_dd* (G_prev.^ 2).* posteri_prev+ ...
            (1-a_dd)* posteri_prime; %a priori SNR
    end

    log_sigma_k= post.* priori./ (1+ priori)- log(1+ priori);    
    vad_decision(i)= sum( log_sigma_k)/ frame_len;    
    if (vad_decision(i)< eta) 
        % noise only frame found
        noise_psd= mu* noise_psd+ (1- mu)* noisy_ps;
        vad( n_start: n_start+ frame_len- 1)= 0;
    else
        vad( n_start: n_start+ frame_len- 1)= 1;
    end
    % === end of VAD ===
    
    G= ( priori./ (1+ priori)).^0.5; % gain function
   
    enh= ifft( noisy_fft.* G, frame_len);
        
    if (i== 1)
        enhanced_speech( n_start: n_start+ frame_len/2- 1)= ...
            enh( 1: frame_len/2);
    else
        enhanced_speech( n_start: n_start+ frame_len/2- 1)= ...
            overlap+ enh( 1: frame_len/2);  
    end
    
    overlap= enh( frame_len/ 2+ 1: frame_len);
    n_start= n_start+ frame_len/2; 
    
    G_prev=G; 
    posteri_prev= post;
    
end

enhanced_speech( n_start: n_start+ frame_len/2- 1)= overlap; 

end
