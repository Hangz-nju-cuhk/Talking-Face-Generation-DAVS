function [ CC, FBE, frames ] = runmfcc( speech, opt )

    N = opt.C ;
    hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));
    [ CC, FBE, frames ] = mfcc( speech, opt.fs, opt.Tw, opt.Ts, opt.alpha, hamming, opt.R, opt.M, N, opt.L );

end