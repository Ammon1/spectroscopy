# Renishaw ploting
Program written to analyze Raman maps done on Renishaw spectrometer (program reads ONLY Renishaw maps export to txt or csv). If someone will send me spectra from other spectrometers I'll upgrade program to read it.

Libraries used:
tkinter 8.6.8
pandas 24.2
numpy 1.16.4
sklearn
scipy 1.2.1
matplotlib 3.1.0

![menu](https://user-images.githubusercontent.com/10612928/61782121-cda93a00-ae05-11e9-8fce-52add35eaba1.png)

To open reference use file->open reference and use file->open data. Data will be uploaded and the first spectrum of the map and the reference will be shown.

![spectra](https://user-images.githubusercontent.com/10612928/61782762-ea923d00-ae06-11e9-8313-fc7621ddd8fa.png)

Next You can use file->clean data to remove spikes and noise from spectra. Spike removing algorithm is based on pandas library and averaging specra. 

![cleaning](https://user-images.githubusercontent.com/10612928/61782863-1f05f900-ae07-11e9-8a56-53114173fdf7.png)

To aply proper substrate subtraction first rescale refrence to data by file->scale reference. Rescaling is used because in other case the subtracting factor may to too large. It is also much more effective to first cale data and than find subtracting factor- this value should be close to 1.

![rescale_ref](https://user-images.githubusercontent.com/10612928/61783068-72784700-ae07-11e9-9bfb-c3f963e13c9b.png)

Subtructing works only for graphene on SiC and only with presented spectral range. Main idea is to find minimum value of integral of spectra in the range of unwonted part. When integral is smallest, this means subtructing was optimal.

After it subtract substrate and You may fit all carbon bands. 

![fitting](https://user-images.githubusercontent.com/10612928/61782377-38f30c00-ae06-11e9-9bd2-ec6eb1caa218.png)

You will obtain maps of FWHM and positions of chosen band.

![maps](https://user-images.githubusercontent.com/10612928/61782962-45c42f80-ae07-11e9-9110-1aaf7a96bb53.png)

On each time consuming stage You have stage bar to inform about the process.

![stage_view](https://user-images.githubusercontent.com/10612928/61783362-f5999d00-ae07-11e9-8bd0-0fc6e40a2420.png)

