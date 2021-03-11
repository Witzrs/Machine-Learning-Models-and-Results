import numpy as np
import os, glob
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks
from math import floor
from PIL import Image, ImageChops
""" Based on code from https://stackoverflow.com/a/49157454"""

""" short time fourier transform of audio signal """ ## Not changed
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    # print(samples)
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    ## Not Changed
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def create_spectrogram(samples, binsize = 2**8, plotpath=None, samplerate=0, colormap="Greys", fig_height=5.82, fig_width=1.93):
    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)


    fig = plt.figure(figsize=(fig_height, fig_width ), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])####
    ax.set_axis_off()                   #### This removes the white borders around the spectrogram, where the axes scales would be shown
    fig.add_axes(ax)                    #### https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content

    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")

    if plotpath: #To save the generated spectrogram to file
        plt.savefig(plotpath)#, bbox_inches="tight")
        plt.close('all')
        plt.clf()
    else: #In case that the user only wants to see the figure and not save it to file 
        plt.show()                  
        plt.close('all')           
        plt.clf()                   

    return ims


def plotstft(audiopath, binsize=2**8, plotpath=None, colormap="Greys", step=0.5, window=1, fig_height=5.82, fig_width=1.93):
    """
    

    """
    samplerate, samples = wav.read(audiopath)
    # i: used for numbering the spectrograms of split parts of the audio
    i=0 
    #initializing the ammount of remaining samples (initially it is equalsamples) 
    remaining_samples = samples
    # print("\n Number of samples: ",len(remaining_samples), "\nsamplerate: ",samplerate, "\nlength: ", len(samples)/samplerate, "\n 1 second sample: ", samples[0:samplerate], "\n")

    while (len(remaining_samples)-(samplerate * step) > step * samplerate):
        imagepath = plotpath + '_' + str(i) + '.png'
        ims = create_spectrogram(remaining_samples[0:samplerate*window], binsize=binsize, plotpath=imagepath, samplerate=samplerate, colormap=colormap, fig_height=fig_height, fig_width=fig_width)
        remaining_samples = remaining_samples[floor(samplerate * window * step):]
        i+=1
    #when the remaining of the file duration is below the defined window, a spectrogram is generated with the rest of the available samples.
    imagepath = plotpath + '_' + str(i) + '.png'
    ims = create_spectrogram(remaining_samples, binsize=binsize, plotpath=imagepath, samplerate=samplerate, colormap=colormap)
    
    return ims

#path to the audio files
inward = os.path.dirname(__file__)

# output_path = 'C:\\Users\\Nat\\OneDrive - Universidade de Lisboa\\Documentos\\Tese\\Spectrograms\\Wide\\'
outward = 'C:\\Users\\Witzr\\Desktop\\spectrograms1\\'

def generate_from_folder(input_path= inward, output_path = outward):
    ims=0.0
    input_path = os.path.join(input_path, 'Recordings')
    for folder  in os.listdir(input_path):
        # print("root: ",root, "\nFolder: ", os.listdir(root))

        specie_name = os.path.basename(folder).replace(" ","_")
        
        for fileX in os.listdir(os.path.join(input_path, folder)): 
            filepath = os.path.join(input_path, folder, fileX)
            filename = (fileX.split('.')[0])
            print("\nSpecie: ", specie_name, "\nAudio file: ", filename)
            plotpath = os.path.join( output_path, specie_name) + '_' + filename
            # try:
            ims = plotstft(filepath, plotpath=plotpath)
            # except:
            #     print("\n Specie: ", specie_name, "\nFile: ", filename)
    return ims
generate_from_folder()
# result = run (input_path, output_path)

def generate_from_file(input_path, output_path, filename):
    ims=0.0
    filepath = input_path
    # print("\nSpecie: ", specie_name, "\nAudio file: ", filename)
    plotpath = output_path + '_' + filename
    # try:
    ims = plotstft(filepath, plotpath=plotpath)
    # except:
    #     print("\n Specie: ", specie_name, "\nFile: ", filename)
    return ims
