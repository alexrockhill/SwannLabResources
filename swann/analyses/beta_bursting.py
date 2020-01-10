import numpy as np
import os.path as op

from pandas import DataFrame

from swann.utils import (get_config, derivative_fname, read_raw,
                         get_events)
from swann.preprocessing import (get_bads, set_bads, get_ica,
                                 preproc_slowfast, slowfast_group,
                                 get_aux_epochs, set_ica_components)

from mne.time_frequency.tfr import cwt
from mne.time_frequency import morlet


def beta_bursting(rawf, event, lfreq=15, hfreq=29, dfreq=1, n_cycles=7,
                  overwrite=False):
    ''' Find beta bursts (15-29 Hz > 6 * median of all signal).
    Parameters
    ----------
    rawf : BIDSDataFile
        The ephys data file from the pybids layout object.
    event : str
        The name of the event in params.json event_id to use.
    lfreq : float
        The lowest frequency to use.
    hfreq : float
        The greatest frequency to use.
    dfreq : int
        The step size between lfreq and hfreq.
    n_cycles : int, np.array
        The number of cycles to use in the Morlet transform
    '''
    beta_burstingf = derivative_fname(rawf, '%s_beta_bursting' % event, 'tsv')
    if op.isfile(beta_burstingf) and not overwrite:
        print('Beta bursts already calculated for %s, ' % rawf.path +
              'use `overwrite=True` to recompute')
        return
    beta_bursts = dict(burst_start=[], burst_duration=[])
    frequencies = np.arange(lfreq, hfreq + dfreq, dfreq)
    raw = read_raw(rawf.path)
    raw.info['bads'] += [ch for ch in get_bads(rawf) if
                         ch not in raw.info['bads']]
    events = get_events(raw, event)
    # raw = apply_ica(rawf)


    DataFrame(beta_bursts).to_csv(beta_bursts, index=False, sep='\t')


'''
Adapted to be identical to the following code from
https://www.biorxiv.org/content/10.1101/644682v1

%% beta event detection
% settings
frequencies = 15:29;
cycles = 7;
mediancutoff = 6;
% preassign
GO = nan(length(chanlocs),length(fname)); SS = GO; FS = GO; BL = GO;
% go through subject
for is = 1:length(fname)
    % clear output
    clear trialevs blevs
    % load
    EEG = pop_loadset(fullfile(csdfolder,[fname{is}(1:end-4) '-csd.set']));
    % get SSRT
    behav = Basicstop_analyze(EEG.behavior);
    SSRT = behav.RT.SSRTi; % integration ssrt
    RT = behav.RT.corgo;
    % look up chans
    [in,b,c] = intersect({EEG.chanlocs.labels},{chanlocs.labels},'stable');
    % create wavelet
    time = -1:1/EEG.srate:1;
    % pre assign
    wavelet_conv_data = zeros(length(frequencies),EEG.pnts);
    % save original EEG
    tEEG = EEG;
    % go through channels
    for ic = 1:EEG.nbchan
        % preassign
        tEEG = EEG;
        % go through frequencies
        for ifr = 1:length(frequencies)
            % freqs and cycles
            f = frequencies(ifr);
            s = cycles/(2*pi*f); 
            % sine and gaussian
            sine_wave = exp(1i*2*pi*f.*time);
            gaussian_win = exp(-time.^2./(2*s^2));
            % normalization factor
            normalization_factor = 1 / (s * sqrt(2* pi));
            % make wavelet
            wavelet = normalization_factor .* sine_wave .* gaussian_win;
            halfwaveletsize = ceil(length(wavelet)/2); % half of the wavelet size
            % convolve with data
            n_conv = length(wavelet) + EEG.pnts - 1; % compute Gaussian
            % fft
            fft_w = fft(wavelet,n_conv);
            fft_e = fft(EEG.data(ic,:),n_conv);
            ift   = ifft(fft_e.*fft_w,n_conv);
            wavelet_conv_data(ifr,:) = abs(ift(halfwaveletsize:end-halfwaveletsize+1)).^2;
        end
        % insert
        tEEG.data = wavelet_conv_data;
        tEEG.nbchan = 1:length(frequencies); tEEG.chanlocs = []; tEEG.icaweights = []; tEEG.icawinv = [];
        % epoch
        eEEG = pop_epoch(tEEG,{'S200' 'S  1' 'S  2'},window);
        epochs = eEEG.data;
        eventsample = abs(window(1)*EEG.srate); % onset of event within epoch
        % get trial info
        accuracy = zeros(eEEG.trials,1); for ie = 1:eEEG.trials; accuracy(ie) = eEEG.epoch(ie).eventacc{1}; end
        ssd = zeros(eEEG.trials,1); for ie = 1:eEEG.trials; ssd(ie) = eEEG.epoch(ie).eventbehav{1}(4+eEEG.epoch(ie).eventbehav{1}(3)); end
        side = zeros(eEEG.trials,1); for ie = 1:eEEG.trials; side(ie) = eEEG.epoch(ie).eventbehav{1}(3); end
        tnum = zeros(eEEG.trials,1); for ie = 1:eEEG.trials; tnum(ie) = eEEG.epoch(ie).eventnr{1}; end
        % convert go-locked stop-trial accuracy to be able to separate stop from go-locked data based on acc variable alone
        tt = {}; for ie = 1:eEEG.trials; tt(ie) = eEEG.epoch(ie).eventtype(find(cell2mat(eEEG.epoch(ie).eventlatency)==0)); end
        accuracy(strcmpi(tt,'S  2')) = accuracy(strcmpi(tt,'S  2')) + 1000; % now all go-locked stop trials will be accuracy+1000
        % get frequency medians
        fmedian = zeros(size(epochs,1),1);
        for ifr = 1:size(epochs,1);
            alldata = [reshape(squeeze(epochs(ifr,:,accuracy<1000)),size(epochs,2)*sum(accuracy<1000),1)];
            fmedian(ifr,1) = median(alldata);
        end
        % go through trials
        if ic == 1; eventnums = zeros(EEG.nbchan,size(eEEG.data,1)); bl_eventnums = eventnums; end % preassign
        for it = 1:size(epochs,3)
            % POST-event
            % get trial TF data (from event to SSRT)
            tdata = squeeze(epochs(:,eventsample+1:end,it));
            % find local maxima
            [peakF,peakT] = find(imregionalmax(tdata));
            % get power for events
            peakpower = zeros(length(peakF),4); % column 1 = value, column 2 =  above threshold
            for ie = 1:length(peakF);
                peakpower(ie,1) = tdata(peakF(ie),peakT(ie));
                if peakpower(ie,1) > mediancutoff*fmedian(peakF(ie)); peakpower(ie,2) = 1; end
                peakpower(ie,3) = peakF(ie)+frequencies(1)-1; % frequency of blob
                peakpower(ie,4) = peakT(ie)*1000/EEG.srate; % ms of blob
            end
            eventnums(ic,it) = sum(peakpower(:,2));
            peakpower = peakpower(logical(peakpower(:,2)),:);
            % store
            if eventnums(ic,it) > 0
                eval(['trialevs.' EEG.chanlocs(ic).labels '.t' num2str(it) ' = [' num2str(peakpower(:,4)') '];']);
            else eval(['trialevs.' EEG.chanlocs(ic).labels '.t' num2str(it) ' = [];']);
            end
            % baseline
            % get trial TF data (from -SSRT to event)
            tdata = squeeze(epochs(:,1:eventsample-1,it));
            % find local maxima
            [peakF,peakT] = find(imregionalmax(tdata));
            % get power for events
            peakpower = zeros(length(peakF),4); % column 1 = value, column 2 =  above threshold
            for ie = 1:length(peakF);
                peakpower(ie,1) = tdata(peakF(ie),peakT(ie));
                peakpower(ie,3) = peakF(ie)+frequencies(1)-1; % frequency of blob
                peakpower(ie,4) = peakT(ie)*1000/EEG.srate; % ms of blob
                if peakpower(ie,1) > mediancutoff*fmedian(peakF(ie)); peakpower(ie,2) = 1; end
            end
            bl_eventnums(ic,it) = sum(peakpower(:,2));
            peakpower = peakpower(logical(peakpower(:,2)),:);
            % store
            if bl_eventnums(ic,it) > 0
                eval(['blevs.' EEG.chanlocs(ic).labels '.t' num2str(it) ' = [' num2str(peakpower(:,4)') '];']);
            else eval(['blevs.' EEG.chanlocs(ic).labels '.t' num2str(it) ' = [];']);
            end
        end
        % store fcz data (from channel identified below)
        if strcmpi(EEG.chanlocs(ic).labels,'FCz'); FCZ = eEEG; end
        if strcmpi(EEG.chanlocs(ic).labels,'C3'); C3 = eEEG; end
        if strcmpi(EEG.chanlocs(ic).labels,'C4'); C4 = eEEG; end
    end
    % trials
    GO(c,is) = mean(eventnums(b,accuracy==1),2);
    SS(c,is) = mean(eventnums(b,accuracy==3),2);
    FS(c,is) = mean(eventnums(b,accuracy==4),2);
    BL(c,is) = mean(bl_eventnums(b,accuracy==1),2); % pre-stim baseline
    % save trial and baseline evs
    save(fullfile(outfolder,['Subject' num2str(is) '-csd.mat']),'trialevs','blevs','SSRT','accuracy','ssd','side','tnum','RT');
    % save individual events at FCZ, C3, C4
    save(fullfile(itfolder,['Subject' num2str(is) '-csd.mat']),'FCZ','C3','C4','accuracy','side');
end
'''