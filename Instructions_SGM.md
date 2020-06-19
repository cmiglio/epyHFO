# Instructions for running SGM detector

To better understand this instructions, a PAT_Example is provided with the results of the detector already provided
(in results folder)

* Use the environment file epyHFO.yml:
`conda env create --name epyHFO --file environment.yml`
* All subjects should have a unique name (for example, here we have one subject called *PAT_Example*)
* All subjects should have a configuration file, saved as config/subjectName/files_process.csv (see *PAT_Example*)
* files_process.csv contain the list of files that have to be processed.
* - files: is the file name
* - label: is a label that you want to put to the file (in this case is Sleep, because the files are from SWS sleep)
* - ini: is the beginning of the file in seconds (because we previously extracted this data from a bigger EEG file)
* - fi: is the end of the file in seconds
* - f_ini: is the initial frequency of interest to run the detector (commonly 80Hz)
* - f_fin: is the final frequency of interest to run the detector (depending on your sampling rate..)
* Subject data (here is in a matlab structure, but you can adapt the code to open other EEG files) should be saved
in data/subjectName/filename.mat this file should have the following structure to match with *files_process.csv*:
* - *files*_*ini*s_*fi*s_*label*.mat
* Results are saved in csv individually for each .mat file and each range of frequencies of interest and a final
*Features_subjectName.csv* is created with all the extracted for all the events found in all the provided files.
 In this file (see /data_example/results/Features_PAT_example.csv), The probability of being an HFO is provided in the
 last column (named *p0*).
* You can plot events if you set to true the variable p_events (line 109 file functions_sgm.py). This may take a while
as a lot of events may appear. Specially in the 200 to 400 hz band (Is not recommended to detect HFOs in this
band, with a sampling frequency of 1024 Hz, the maximum band to detect should be between sfreq/5 or sfreq/4.
The results in this band are only shown as an example)