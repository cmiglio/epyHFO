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


# 1. Import needed packages
import time
from scipy.io import loadmat
import numpy as np
import mne
import pandas as pd
import functions_sgm as fc
from sklearn.mixture import GaussianMixture
import warnings


def extract_features(patients, results_path, config_path, data_path):
    for patient in patients:
        # Read the configuration file for the current patient
        files = pd.read_csv(config_path + patient + '/files_process.csv', sep=';')
        for index, row in files.iterrows():
            start_time = time.time()
            filename = row['files']
            ini = row['ini']
            fi = row['fi']
            state = row['label']
            f_ini = row['f_ini']
            f_fin = row['f_fin']

            # Load mat file
            matfile = data_path + patient + '/' + filename + '_' + str(ini) + 's-' + str(fi) + 's_' + state + '.mat'
            annots = loadmat(matfile, squeeze_me=True)
            # Extract variables from matfile
            data = annots['data']
            sfreq = annots['sfreq']
            ch_names = annots['ch_names']
            ch_types = annots['ch_types']
            differential = annots['differential']
            average = annots['average']
            needle_names = annots['needle_names']
            needle = annots['needle1']
            pos = annots['pos']

            # 3. Create MNE raw object & info file
            info = mne.create_info(ch_names=ch_names.tolist(), sfreq=sfreq, ch_types=ch_types.tolist())
            raw = mne.io.RawArray(data, info)

            # 4. Filter data above cut_freq
            raw80 = raw.copy()
            raw80.filter(f_ini, f_fin, fir_design='firwin', verbose=False)
            raw80.notch_filter([50, 100, 150, 200, 250], fir_design='firwin', verbose=False)
            raw80.notch_filter(150, fir_design='firwin', verbose=False)

            # 5. Extract events:
            #     - Compute threshold
            #     - Select EOIs
            #     - Compute Stockwell transform
            #     - Detect events above binarization
            #     - Extract features

            # Declare empty lists for all the features
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

            channels = list()
            needles = list()
            needle_name = list()
            x = list()
            y = list()
            z = list()
            th_aux = list()

            # Hilbert transform amplitude
            raw_hilb = raw80.copy()
            raw_hilb.apply_hilbert()

            # Compute threshold
            chan_n = 0
            lsegments = 0.5 * sfreq
            window = np.arange(0, raw80.last_samp, lsegments)
            channels_vector = np.arange(0, len(raw80.ch_names))
            th = np.zeros(channels_vector.shape)

            # Stockwell parameters
            win_size = int(0.5 * sfreq)  # Duration of stockwell window = 1 second

            for chan_n in channels_vector:
                data = raw.get_data(picks=chan_n)
                data_80 = raw80.get_data(picks=chan_n)
                data_hil = np.abs(raw_hilb.get_data(picks=chan_n))
                # Obtain threshold
                th[chan_n], bs = fc.compute_threshold(window, data_80, .95, .999, lsegments)
                # Obtain events of interest
                eoi = fc.compute_eois(data_80, data_hil, th[chan_n], sfreq)
                # Extract features for each event using Stockwell Transform and image processing techniques
                feature_list = fc.compute_features(eoi, data, data_80, data_hil, win_size, f_ini, f_fin,
                                                   sfreq, th[chan_n], bs, ch_names[chan_n], 2, index)
                n_events = len(feature_list[0])
                Ev_ini.extend(feature_list[0])
                Ev_fi.extend(feature_list[1])
                dur_f.extend(feature_list[2])
                dur_t.extend(feature_list[3])
                area.extend(feature_list[4])
                entropy_calc.extend(feature_list[5])
                perimeter.extend(feature_list[6])
                symmetryt.extend(feature_list[7])
                symmetryf.extend(feature_list[8])
                oscillations.extend(feature_list[9])
                kurtosis.extend(feature_list[10])
                skewness.extend(feature_list[11])
                inst_freq.extend(feature_list[12])
                amplitude.extend(feature_list[13])
                amplitude_norm.extend(feature_list[14])
                mean_power.extend(feature_list[15])
                max_power.extend(feature_list[16])

                # Parameters that depend on the channel
                channels.extend([ch_names[chan_n]] * n_events)
                needles.extend([needle[chan_n]] * n_events)
                needle_name.extend([needle_names[needle[chan_n] - 1]] * n_events)
                x.extend([pos[chan_n, 0]] * n_events)
                y.extend([pos[chan_n, 1]] * n_events)
                z.extend([pos[chan_n, 2]] * n_events)
                th_aux.extend([th[chan_n]] * n_events)

            # Parameters that depend on the subject
            subj = [patient] * np.shape(perimeter)[0]
            file = [filename] * np.shape(perimeter)[0]
            iniglob = [ini] * np.shape(perimeter)[0]
            figlob = [fi] * np.shape(perimeter)[0]
            stateglob = [state] * np.shape(perimeter)[0]
            sfreqglob = [sfreq] * np.shape(perimeter)[0]
            differentialglob = [differential] * np.shape(perimeter)[0]
            averageglob = [average] * np.shape(perimeter)[0]

            list_all = list(zip(subj, file, iniglob, figlob, stateglob, sfreqglob,
                                channels, differentialglob, averageglob, needles, needle_name,
                                x, y, z, th_aux, Ev_ini, Ev_fi, dur_f, dur_t, area, entropy_calc,
                                perimeter, symmetryt, symmetryf, oscillations,
                                kurtosis, skewness, inst_freq, amplitude, amplitude_norm, mean_power, max_power))

            features = pd.DataFrame(list_all, columns=['Subject', 'Filename', 'Ini', 'Fi', 'State', 'Sfreq',
                                                       'Channel', 'Differential', 'Average', 'Needle', 'Needle name',
                                                       'x', 'y', 'z', 'Threshold', 'Ev ini', 'Ev fi', 'Dur f', 'Dur t',
                                                       'Area', 'Entropy', 'Perimeter', 'Symmetry T',
                                                       'Symmetry F', 'Oscillations', 'Kurtosis',
                                                       'Skewness', 'Inst freq', 'Amplitude', 'Amplitude norm',
                                                       'mean_power', 'max_power'])
            # Some other features
            features['Global ini'] = features['Ini'] + features['Ev ini'] / sfreq
            features['Global end'] = features['Ini'] + features['Ev fi'] / sfreq
            features['f_ini'] = f_ini
            features['f_fin'] = f_fin

            features.to_csv(results_path + 'Features_eois_' + patient + '_' + filename + '_' + str(ini) + '_'
                            + str(fi) + '_' + state + '_' + str(f_ini) + '_' + str(f_fin) + '.csv')

            print(str(np.round(time.time() - start_time)) + ' seconds elapsed', end='\r')
            print('Finished: ' + results_path + 'Features_eois_' + patient + '_' + filename + '_' + str(ini) + '_'
                  + str(fi) + '_' + state + '_' + str(f_ini) + '_' + str(f_fin) + '.csv', end='\r')


def obtain_gmm(patients, results_path, config_path):
    warnings.filterwarnings('ignore')
    for patient in patients:
        files = pd.read_csv(config_path + patient + '/files_process.csv', sep=';')
        first = True
        for index, f in files.iterrows():
            file_name = 'Features_eois_' + patient + '_' + f['files'] + '_' + str(f['ini']) + '_' + str(f['fi']) + '_' + \
                        f['label'] + '_' + str(f['f_ini']) + '_' + str(f['f_fin']) + '.csv'
            if first:
                first = False
                features = pd.read_csv(str(results_path + file_name),
                                       usecols=['Subject', 'Filename', 'Ini', 'Fi', 'Global ini', 'Global end', 'State',
                                                'Sfreq', 'Channel', 'Differential', 'Average', 'Needle', 'Needle name',
                                                'x', 'y', 'z', 'Threshold', 'Ev ini', 'Ev fi', 'Dur f', 'Dur t',
                                                'Area', 'Entropy', 'Perimeter', 'Symmetry T', 'Symmetry F',
                                                'Oscillations', 'Kurtosis', 'Skewness', 'Inst freq', 'Amplitude',
                                                'Amplitude norm', 'f_ini', 'f_fin', 'mean_power', 'max_power'])

            else:
                to_append = pd.read_csv(str(results_path + file_name),
                                        usecols=['Subject', 'Filename', 'Ini', 'Fi', 'Global ini', 'Global end',
                                                 'State',
                                                 'Sfreq', 'Channel', 'Differential', 'Average', 'Needle', 'Needle name',
                                                 'x', 'y', 'z', 'Threshold', 'Ev ini', 'Ev fi', 'Dur f', 'Dur t',
                                                 'Area', 'Entropy', 'Perimeter', 'Symmetry T', 'Symmetry F',
                                                 'Oscillations', 'Kurtosis', 'Skewness', 'Inst freq', 'Amplitude',
                                                 'Amplitude norm', 'f_ini', 'f_fin','mean_power', 'max_power'])

                features = features.append(to_append, ignore_index=True)

        gmm = GaussianMixture(n_components=3)
        features['LogArea'] = np.log10(features['Area'])
        feature = ['Area']
        featuresclust = features[feature].copy()
        X = featuresclust.values
        gmm.fit(X)
        group = gmm.predict(X)
        prob = gmm.predict_proba(X)
        means = gmm.means_
        hfo_group = np.argsort(means[:, 0])[1]
        hfo_prob = prob[:, hfo_group]
        features['group'] = group
        features['p0'] = hfo_prob

        features.to_csv(results_path + 'Features_' + patient + '.csv')
