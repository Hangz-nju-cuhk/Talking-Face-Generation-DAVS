function savemfcc(file_name, save_dir)
        opt.fs = 16000;
        opt.Tw = 25;
        opt.Ts = 10;
        opt.alpha = 0.97;
        opt.R = [300 3700];
        opt.M = 13;
        opt.C = 13;
        opt.L = 22;

        [Speech, fs] = audioread(file_name);
        [length_of_speech, channel] = size(Speech);
        if channel == 2
            Speech = (Speech(:, 1));
        end
        
        [ MFCCs, ~, ~ ] = runmfcc( Speech, opt );
        mfccs = MFCCs(2:end, :);
        num_bins = floor(length_of_speech / fs * 25);
        for l = 2:num_bins - 4
            save_mfcc20 = mfccs(:, 4 * l -7  : 4 * l + 19 -7);

            f2 = fopen(fullfile(save_dir, [num2str(l), '.bin']), 'wb');
            fwrite(f2, save_mfcc20, 'double');
            fclose(f2);                    
        end
