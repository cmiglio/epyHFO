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

import sgm as sgm
# Load files

# Define which patients will be used
patients = ['PAT_example']
results_path = 'data_example/results/'
config_path = 'data_example/config/'
data_path = 'data_example/data/'

sgm.extract_features(patients, results_path, config_path, data_path)
sgm.obtain_gmm(patients, results_path, config_path)
