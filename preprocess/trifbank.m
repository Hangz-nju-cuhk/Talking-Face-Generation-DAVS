function [ H, f, c ] = trifbank( M, K, R, fs, h2w, w2h )
% TRIFBANK Triangular filterbank.
%
%   [H,F,C]=TRIFBANK(M,K,R,FS,H2W,W2H) returns matrix of M triangular filters 
%   (one per row), each K coefficients long along with a K coefficient long 
%   frequency vector F and M+2 coefficient long cutoff frequency vector C. 
%   The triangular filters are between limits given in R (Hz) and are 
%   uniformly spaced on a warped scale defined by forward (H2W) and backward 
%   (W2H) warping functions.
%
%   Inputs
%           M is the number of filters, i.e., number of rows of H
%
%           K is the length of frequency response of each filter 
%             i.e., number of columns of H
%
%           R is a two element vector that specifies frequency limits (Hz), 
%             i.e., R = [ low_frequency high_frequency ];
%
%           FS is the sampling frequency (Hz)
%
%           H2W is a Hertz scale to warped scale function handle
%
%           W2H is a wared scale to Hertz scale function handle
%
%   Outputs
%           H is a M by K triangular filterbank matrix (one filter per row)
%
%           F is a frequency vector (Hz) of 1xK dimension
%
%           C is a vector of filter cutoff frequencies (Hz), 
%             note that C(2:end) also represents filter center frequencies,
%             and the dimension of C is 1x(M+2)
%
%   Example
%           fs = 16000;               % sampling frequency (Hz)
%           nfft = 2^12;              % fft size (number of frequency bins)
%           K = nfft/2+1;             % length of each filter
%           M = 23;                   % number of filters
%
%           hz2mel = @(hz)(1127*log(1+hz/700)); % Hertz to mel warping function
%           mel2hz = @(mel)(700*exp(mel/1127)-700); % mel to Hertz warping function
%
%           % Design mel filterbank of M filters each K coefficients long,
%           % filters are uniformly spaced on the mel scale between 0 and Fs/2 Hz
%           [ H1, freq ] = trifbank( M, K, [0 fs/2], fs, hz2mel, mel2hz );
%
%           % Design mel filterbank of M filters each K coefficients long,
%           % filters are uniformly spaced on the mel scale between 300 and 3750 Hz
%           [ H2, freq ] = trifbank( M, K, [300 3750], fs, hz2mel, mel2hz );
%
%           % Design mel filterbank of 18 filters each K coefficients long, 
%           % filters are uniformly spaced on the Hertz scale between 4 and 6 kHz
%           [ H3, freq ] = trifbank( 18, K, [4 6]*1E3, fs, @(h)(h), @(h)(h) );
%
%            hfig = figure('Position', [25 100 800 600], 'PaperPositionMode', ...
%                              'auto', 'Visible', 'on', 'color', 'w'); hold on; 
%           subplot( 3,1,1 ); 
%           plot( freq, H1 );
%           xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 
%       
%           subplot( 3,1,2 );
%           plot( freq, H2 );
%           xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 
%       
%           subplot( 3,1,3 ); 
%           plot( freq, H3 );
%           xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 
%
%   Reference
%           [1] Huang, X., Acero, A., Hon, H., 2001. Spoken Language Processing: 
%               A guide to theory, algorithm, and system development. 
%               Prentice Hall, Upper Saddle River, NJ, USA (pp. 314-315).

%   Author  Kamil Wojcicki, UTD, June 2011


    if( nargin~= 6 ), help trifbank; return; end; % very lite input validation

    f_min = 0;          % filter coefficients start at this frequency (Hz)
    f_low = R(1);       % lower cutoff frequency (Hz) for the filterbank 
    f_high = R(2);      % upper cutoff frequency (Hz) for the filterbank 
    f_max = 0.5*fs;     % filter coefficients end at this frequency (Hz)
    f = linspace( f_min, f_max, K ); % frequency range (Hz), size 1xK
    fw = h2w( f );

    % filter cutoff frequencies (Hz) for all filters, size 1x(M+2)
    c = w2h( h2w(f_low)+[0:M+1]*((h2w(f_high)-h2w(f_low))/(M+1)) );
    cw = h2w( c );

    H = zeros( M, K );                  % zero otherwise
    for m = 1:M 

        % implements Eq. (6.140) on page 314 of [1] 
        % k = f>=c(m)&f<=c(m+1); % up-slope
        % H(m,k) = 2*(f(k)-c(m)) / ((c(m+2)-c(m))*(c(m+1)-c(m)));
        % k = f>=c(m+1)&f<=c(m+2); % down-slope
        % H(m,k) = 2*(c(m+2)-f(k)) / ((c(m+2)-c(m))*(c(m+2)-c(m+1)));

        % implements Eq. (6.141) on page 315 of [1]
        k = f>=c(m)&f<=c(m+1); % up-slope
        H(m,k) = (f(k)-c(m))/(c(m+1)-c(m));
        k = f>=c(m+1)&f<=c(m+2); % down-slope
        H(m,k) = (c(m+2)-f(k))/(c(m+2)-c(m+1));
       
   end

   % H = H./repmat(max(H,[],2),1,K);  % normalize to unit height (inherently done)
   % H = H./repmat(trapz(f,H,2),1,K); % normalize to unit area 


% EOF
