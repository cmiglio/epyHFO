# epyHFO is an open-source Python package for automatic detection of high frequency oscillations (HFOs) in EEG
# recordings. If you use the source code, please make sure to reference both the package and the paper:
# * Migliorelli, C., Bachiller, A., Alonso, JF., Romero S., Aparicio J., Jacobs-Le Van, J., Mañanas, MA.,
#   San Antonio-Arce V. (Apr 2020). SGM: a novel time-frequency algorithm based on unsupervised learning improves
#   high-frequency oscillation detection in epilepsy. J Neural Eng. 22;17(2):026032. doi: 10.1088/1741-2552/ab8345.
# * Migliorelli, C. (2020). epyHFO v1.0, https://github.com/cmiglio/epyHFO (Zenodo Doi: https://doi.org/10.5281/zenodo.3894472)
# --------------------------------------------------------------------------------------------------------------------
# Copyright (C) 2020, Centro de investigación Biomedica en red (CIBER-BBN), Universitat Politecnica de Catalunya (UPC)
# Author: Carolina Migliorelli
#
# This software is distributed under the terms of the BSD 3-Clause License licence
#
#
# FOR RESEARCH PURPOSES ONLY. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
# AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# --------------------------------------------------------------------------------------------------------------------


from scipy.stats import entropy
import numpy as np
from mne.time_frequency import tfr_array_stockwell
from skimage.morphology import reconstruction
from skimage.filters import threshold_yen
from skimage.measure import label, regionprops, shannon_entropy
from scipy import signal, stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2):]


def compute_threshold(window, data_80, ent_q, bs_q, lsegments):
    ent = np.zeros(window.shape)
    # ent_q : quantile for entropy of the autocorrelation (usually 95%)
    # bs_q: quantile for the baseline (usually 99.9%)
    for n, W in enumerate(window):
        # Compute the autocorrelation of the entropy for each segment
        acor = np.abs(autocorr(data_80[0, int(W):int(W + lsegments)]))
        ent[n] = entropy(acor)
    segments_ok = window[ent < np.quantile(ent, ent_q)]
    segments_ok = segments_ok.astype('int')
    bs = np.abs(data_80[0, segments_ok[0]:segments_ok[0] + int(lsegments)])
    for n in np.arange(1, segments_ok.shape[0]):
        bs = np.append(bs, np.abs(data_80[0, segments_ok[n]:segments_ok[n] + int(lsegments)]))
    th = np.quantile(bs, bs_q)
    return th, bs


def compute_eois(data_80, data_hil, th, sfreq):
    change = np.diff((data_hil[0, :] > th * 0.5) * 1)
    if change[np.where(change)[0][0]] == -1:
        change[np.where(change)[0][0]] = 0
    ch_pos = np.sum(change == 1)
    ch_neg = np.sum(change == -1)
    if ch_pos > ch_neg:
        p_aux_ini = np.where(change == 1)[0][:-2]
        p_aux_fin = np.where(change == -1)[0]
    elif ch_pos < ch_neg:
        p_aux_ini = np.where(change == 1)[0]
        p_aux_fin = np.where(change == -1)[0][1:]
    elif ch_pos == ch_neg:
        p_aux_ini = np.where(change == 1)[0]
        p_aux_fin = np.where(change == -1)[0]
    eoi = np.array(list(zip(p_aux_ini, p_aux_fin)))
    # Discard first and last second (compute stockwell transform)
    eoi = eoi[eoi[:, 0] > sfreq * 1, :]
    eoi = eoi[eoi[:, 1] < data_80.shape[1] - sfreq * 1, :]
    eoi = eoi[(np.diff(eoi) > 6 * sfreq / 1000)[:, 0], :]  # at least 6ms
    eoi_ok = np.zeros(0)
    for e in eoi:  # We clear all the eois that any point is above threshold
        if np.sum(np.abs(data_80[0, (e[0]):(e[1])]) > th) != 0:
            eoi_ok = np.append(eoi_ok, e)
    eoi_ok = (eoi_ok.astype(int))
    eoi = np.reshape(eoi_ok, [int(eoi_ok.shape[0] / 2), 2])
    difference = np.diff(eoi_ok)
    a = np.where(difference < 10 * sfreq / 1000)
    odds = np.where(a[0].astype(int) % 2 != 0)
    inicios = np.asarray([a[0][odds]]) - 1
    finales = np.asarray([a[0][odds]]) + 2
    a2 = np.where(difference >= 10 * sfreq / 1000)
    odds = np.where(a2[0].astype(int) % 2 != 0)
    inicios_ok = np.asarray([a2[0][odds]]) - 1
    finales_ok = np.asarray([a2[0][odds]])
    inicios = np.sort(np.concatenate((inicios[0], inicios_ok[0])))
    finales = np.sort(np.concatenate((finales[0], finales_ok[0])))
    u, indices = np.unique(finales, return_index=True, return_inverse=False, return_counts=False, axis=None)
    indices_actualizados = np.sort(np.concatenate((inicios[indices], finales[indices])))
    eoi_ok = eoi_ok[indices_actualizados]
    eoi = np.reshape(eoi_ok, [int(eoi_ok.shape[0] / 2), 2])
    return eoi


def compute_features(eoi, data, data_80, data_hil, win_size, f_ini, f_fin, sfreq, th, bs, channel, version, file_id):
    epoch = np.zeros([1, 1, win_size * 2])
    p_events = False
    Ev_ini = list()
    Ev_fi = list()
    dur_f = list()
    dur_t = list()
    area = list()
    entropy_calc = list()
    perimeter = list()
    symmetryt = list()
    symmetryf = list()
    oscillations = list()
    kurtosis = list()
    skewness = list()
    inst_freq = list()
    amplitude = list()
    amplitude_norm = list()
    mean_power = list()
    max_power = list()
    for idx, event in enumerate(eoi):
        # Compute Stockwell transform for each eoi, binarize and obtain event features
        mid_event = ((event[1] + event[0]) / 2).astype(int)
        epoch[0, 0, :] = data[:, mid_event - win_size:mid_event + win_size]
        pwr = tfr_array_stockwell(epoch, fmin=0, fmax=sfreq/2, width=3, sfreq=sfreq,
                                  n_fft=int(win_size * 2))
        single_epoch = pwr[0][0, f_ini:f_fin, int(win_size / 4):-int(win_size / 4)]
        single_epoch_all = pwr[0][0, 0:f_fin, int(win_size / 4):-int(win_size / 4)]
        frec_vector = pwr[2]
        # Extract properties from pwr "image"

        image = single_epoch.astype('float32')
        image_all = single_epoch_all.astype('float32')
        if version == 1:
            # H-dome correction
            seed = np.copy(image)
            seed[1:-1, 1:-1] = image.min()
            mask = image
            dilated = reconstruction(seed, mask, method='dilation')
            image_corrected = image - dilated
            # Thresholding with yet
            thresh = threshold_yen(image_corrected)
            binary = image_all > thresh

        if version == 2:
            t_mig = int(single_epoch.shape[1] / 2)
            thresh = np.quantile(single_epoch[:, t_mig], 0.75)
            binary = image_all > thresh
        label_image = label(binary)
        # Extract features
        n_events = 0
        for idx2, region in enumerate(regionprops(label_image)):
            minf, mint, maxf, maxt = region.bbox
            # Just consider events that its middle frequency is higher or equal than f_ini
            mid_f = (frec_vector[minf] + frec_vector[maxf]) / 2
            # Just use events that are in the middle of the st (because is centered in the eoi)
            if (single_epoch.shape[1] / 2 >= mint) & (single_epoch.shape[1] / 2 <= maxt) & (mid_f >= f_ini):
                n_events = n_events + 1
                # Global times
                Ev_ini.append(event[0])
                Ev_fi.append(event[1])
                # Duration in frequency and time
                dur_f.append(maxf - minf)
                dur_t.append(maxt - mint)
                # Area
                area.append(region.area)
                # Entropy
                aux = np.zeros(image_all.shape)
                aux[minf:maxf, mint:maxt] = image_all[minf:maxf, mint:maxt]
                entropy_calc.append(shannon_entropy(aux))
                # Other features from regionpropos
                perimeter.append(region.perimeter)
                # Symmetry
                mean_power.append(np.mean(image_all[minf:maxf, mint:maxt]))
                max_power.append(np.max(image_all[minf:maxf, mint:maxt]))

                hfo = data_80[0, mid_event - int(3 * win_size / 4) + mint - 10: mid_event - int(
                    3 * win_size / 4) + maxt + 10]
                hfo_hil = data_hil[0, mid_event - int(3 * win_size / 4) + mint - 10: mid_event - int(
                    3 * win_size / 4) + maxt + 10]
                max_id = np.where(image_all == np.max(image_all[minf:maxf, mint:maxt]))

                symt = (max_id[1][0] - mint) / (maxt - mint)
                symf = (max_id[0][0] - minf) / (maxf - minf)

                symmetryt.append(symt)
                symmetryf.append(symf)
                # number of oscillations (number of peaks)
                peaks = signal.find_peaks(hfo)
                hil_th = (hfo_hil > th / 2)
                ispeak = np.zeros(hfo.shape, dtype=bool)
                ispeak[peaks[0]] = True
                oscillations.append(np.sum(ispeak * hil_th))

                # Kurtosis and skewness (from hilbert transform)

                kurtosis.append(stats.kurtosis(hfo_hil))
                skewness.append(stats.skew(hfo_hil))
                amplitude.append(np.max(hfo_hil))
                amplitude_norm.append(np.max(hfo_hil) / np.mean(bs))
                # Mean peak frequency and std peak frequency
                peaks = signal.find_peaks(np.abs(hfo))
                inst_freq.append(1 / np.mean(np.diff(peaks[0])) * sfreq * 0.5)

                if p_events == True:
                    ev = pd.DataFrame([[mint, maxt, minf, maxf, region.area, shannon_entropy(aux), symt, symf,
                                        np.sum(ispeak * hil_th), 1 / np.mean(np.diff(peaks[0])) * sfreq * 0.5,
                                        np.max(hfo_hil)]], columns=['t_ini', 't_fin', 'f_ini', 'f_fin', 'Area',
                                                                    'Entropy', 'Symmetry T', 'Symmetry F',
                                                                    'Oscillations', 'Inst freq', 'Amplitude'])
                    dat = data[0, mid_event - win_size:mid_event + win_size]
                    datfilt = data_80[0, mid_event - win_size:mid_event + win_size]
                    tf_dat = pwr[0][0, :, int(win_size / 4):-int(win_size / 4)]
                    tf_dat_bin = tf_dat > thresh
                    frec_vector = (pwr[2])
                    image_path = '/home/cmigliorelli/synology/Epilepsia_HSJD/Resultados/hfos_graficas/'
                    plot_eoi(ev, sfreq, dat, datfilt, tf_dat, tf_dat_bin, frec_vector, image_path, channel, idx, idx2,
                             file_id)

    feature_list = [Ev_ini, Ev_fi, dur_f, dur_t, area, entropy_calc, perimeter, symmetryt,
                    symmetryf, oscillations, kurtosis, skewness, inst_freq, amplitude,
                    amplitude_norm, mean_power, max_power]
    return feature_list


def plot_eoi(event, sfreq, dat, datfilt, tf_dat, tf_dat_bin, frec_vector, image_path, channel, idx_plot, idx_plot2,
             file_id):
    filtwin = int(sfreq * 0.5)
    mint = event['t_ini'].values
    maxt = event['t_fin'].values
    minf = event['f_ini'].values
    maxf = event['f_fin'].values
    lowcut = minf
    if maxf >= len(frec_vector):
        highcut = frec_vector[-1] - 1
    else:
        highcut = maxf

    # Figure properties
    sns.set_style("whitegrid")
    fig_size = (16, 12)
    fig, ax = plt.subplots(2, 2, figsize=fig_size)
    plt.tight_layout()
    interval = np.arange(int(filtwin / 2), int(len(dat) - filtwin / 2))

    textstr = '\n'.join((
        r'$Event Index=%.2f$' % (idx_plot,),
        r'$Area=%.2f$' % (event['Area'],),
        r'$Entropy=%.2f$' % (event['Entropy'],),
        r'$Symmetry T=%.2f$' % (event['Symmetry T'],),
        r'$Symmetry F=%.2f$' % (event['Symmetry F'],),
        r'$Oscillations=%.2f$' % (event['Oscillations'],),
        r'$Inst freq=%.2f$' % (event['Inst freq'],),
        r'$Amplitude=%.2f$' % (event['Amplitude'],)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # time plot without filtering
    ax[0, 0].plot(dat[interval])
    ax[0, 0].axvline(x=filtwin / 2, color='r')
    ax[0, 0].set_title('Raw data')
    ax[0, 0].text(0.05, 0.9, textstr, transform=ax[0, 0].transAxes, fontsize=16,
                  verticalalignment='top', bbox=props)
    # time plot with filter filtering
    ax[0, 1].plot(datfilt[interval])
    ax[0, 1].axvline(x=filtwin / 2, color='r')
    ax[0, 1].set_title('Filtered between ' + str(lowcut) + '-' + str(highcut) + ' Hz')

    # Time-frequency stockwell
    sns.heatmap(ax=ax[1, 0], data=tf_dat, robust=True, cbar=False, yticklabels=False, xticklabels=False)
    ax[1, 0].axhline(y=minf, color='lavender', alpha=.6, ls='--')
    ax[1, 0].axhline(y=maxf, color='lavender', alpha=.6, ls='--')
    ax[1, 0].axvline(x=mint, color='lavender', alpha=.6, ls='--')
    ax[1, 0].axvline(x=maxt, color='lavender', alpha=.6, ls='--')
    ax[1, 0].set(ylim=[0, tf_dat.shape[0]])
    ax[1, 0].set_yticks(np.arange(0, tf_dat.shape[0], 80))
    ax[1, 0].set_yticklabels(frec_vector[np.arange(0, tf_dat.shape[0], 80)])
    ax[1, 0].set_xticks(np.arange(0, tf_dat.shape[1], 100))
    ax[1, 0].set_xticklabels(np.arange(0, tf_dat.shape[1], 100))

    # Time-frequency above lowcut
    sns.heatmap(ax=ax[1, 1], data=tf_dat_bin, robust=True, cbar=False, yticklabels=False, xticklabels=False)
    ax[1, 1].axhline(y=minf, color='lavender', alpha=.6, ls='--')
    ax[1, 1].axhline(y=maxf, color='lavender', alpha=.6, ls='--')
    ax[1, 1].axvline(x=mint, color='lavender', alpha=.6, ls='--')
    ax[1, 1].axvline(x=maxt, color='lavender', alpha=.6, ls='--')
    ax[1, 1].set(ylim=[0, tf_dat_bin.shape[0]])
    ax[1, 1].set_yticks(np.arange(0, tf_dat_bin.shape[0], 80))
    ax[1, 1].set_yticklabels(frec_vector[np.arange(0, tf_dat_bin.shape[0], 80)])
    ax[1, 1].set_xticks(np.arange(0, tf_dat_bin.shape[1], 100))
    ax[1, 1].set_xticklabels(np.arange(0, tf_dat_bin.shape[1], 100))

    if not os.path.exists(image_path + channel):
        os.mkdir(image_path + channel)
    fig.savefig(image_path + channel + '/' + str(file_id) + '_' + str(idx_plot) + '_' + str(idx_plot2) + '.png')
    plt.close(fig)
