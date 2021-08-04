import pandas as pd
import matplotlib.pyplot as plt
import nanoscope
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from os.path import dirname, realpath, join
from nanoscope import files
from nanoscope.constants import FORCE, METRIC, VOLTS, PLT_kwargs
from nanoscope.tools import script_tools as st
import math
from scipy.optimize import curve_fit
import os

def data_extract(path):
    with files.ScriptFile(path) as file_:
        channel_num=len(file_)
        seg_info=file_.segments_info

        seg_no=[] #segment No.
        tip_bias=[] #tip bias in each segment
        start_freq=[] #start frequency
        freq_range=[] #segment range, here is frequency range
        seg_samples=[] #samples per segment
        end_freq=[] #end frequency

        for i,info in enumerate(seg_info):
            seg_no.append(i) #segment No.
            tip_bias.append(info.tip_bias) #tip bias in each segment
            start_freq.append(info.ramp_start) #start frequency
            freq_range.append(info.ramp_size) #segment range, here is frequency range
            seg_samples.append(info.size) #samples per segment
            end_freq.append(info.ramp_start+info.ramp_size) #end frequency

    #     freq=np.arange(start_freq[1],start_freq[1]+freq_range[1],freq_range[1]/seg_samples[1])

        input2=[]
        amplitude=[]
        phase=[]
        inphase=[]
        quadrature=[]


        for i in seg_no:
            input2.append(file_[1].get_script_segment_data(i,METRIC))
            amplitude.append(file_[2].get_script_segment_data(i,METRIC))
            phase.append(file_[3].get_script_segment_data(i,METRIC))
            inphase.append(file_[4].get_script_segment_data(i,METRIC))
            quadrature.append(file_[5].get_script_segment_data(i,METRIC))

        input2=np.array(input2)
        amplitude=np.array(amplitude)
        phase=np.array(phase)
        inphase=np.array(inphase)
        quadrature=np.array(quadrature)

    #     start_freqs=np.full(len(seg_no),start_freq)
    #     end_freqs=np.full(len(seg_no),(start_freq+freq_range))
        df=pd.DataFrame({'Segment_No':seg_no,'Samples':seg_samples,'Start_freq':start_freq,'End_freq':end_freq,'Tip_Bias':tip_bias, 'Input2':input2[:,1],'Amplitude':amplitude[:,1],'Phase':phase[:,1],'Inphase':inphase[:,1],'Quadrature':quadrature[:,1],'Time':amplitude[:,0]})
        df=df[1:-1]
    return df


# Create a Frequency array
def single_segment(df,seg_index):
    start_freq=df.loc[seg_index]['Start_freq']
    end_freq=df.loc[seg_index]['End_freq']
    samples=df.loc[seg_index]['Samples']
    freq_arr=np.arange(start_freq,end_freq,(end_freq-start_freq)/samples)
    amp_arr=df.loc[seg_index]['Amplitude']
    phase_arr=df.loc[seg_index]['Phase']

    return [np.array(amp_arr),np.array(phase_arr),np.array(freq_arr)]

# Lorentz model is used to fit the curve
def lorentzmodel(x,a,b,f,q):
    return b + (a * f ** 2 / q) / ((f ** 2 - x ** 2) ** 2 + (f * x / q) ** 2) ** 0.5


def output_fit(df,seg_index):
    [amp_arr,phase_arr,freq_arr]=single_segment(df,seg_index)
    # Do lorentz model and least square fitting here
    # Initial Guess
    a=max(amp_arr)
    b=1
    f=freq_arr[np.argmax(amp_arr)]
    q=50
    initial_guess=[a,b,f,q]

    a0=[]
    f0=[]
    Q=[]

    # curve_fit from scipy is used here to optimize the parameters
    # failed fitting is printed and np.NaN is used to fill into the array
    try:
        popt,pcov=curve_fit(lorentzmodel,freq_arr,amp_arr,initial_guess)
        a0=popt[0]
        b=popt[1]
        f0=popt[2]
        Q=popt[3]
    except:
        a0=np.NaN
        b=np.NaN
        f0=np.NaN
        Q=np.NaN
        print('Fitting not Found')
    #Determine sign of max/min inphase
    inphase_arr=df.loc[seg_index]['Inphase']
    abs_inphase_arr=abs(inphase_arr)
    index=np.argmax(np.array(abs_inphase_arr))
    sign_inphase=(inphase_arr[index]/abs_inphase_arr[index])
    return a0,f0,Q,sign_inphase


def fit_and_clean_table(df,freq_index):
    # user determine the sample index for single frequency
    a0=[]
    f0=[]
    q=[]
    sign_inphase=[]
    amp_single_freq=[]
    phase_single_freq=[]
    #Determine the freq
    single_freq=df.iloc[0]['Start_freq']+(df.iloc[0]['End_freq']-df.iloc[0]['Start_freq'])/df.iloc[0]['Samples']*freq_index

    amp_array=[]
    freq_array=[]

    for i in df.index:
        # get fitting results
        fit_data=output_fit(df,i)
        a0.append(fit_data[0])
        f0.append(fit_data[1])
        q.append(fit_data[2])
        sign_inphase.append(fit_data[3])
        #Determine the amplitude and phase at single freq
        amp_single_freq.append(df.loc[i]['Amplitude'][freq_index])
        phase_single_freq.append(df.loc[i]['Phase'][freq_index])


        # Adding frequency and amplitude arrays in each index for further plots
        [amp_arr,phase_arr,freq_arr]=single_segment(df,i)
        amp_array.append(amp_arr)
        freq_array.append(freq_arr)

    # Add fitted Amplitude, f0 and Q into dataframe
    df['A0']=a0
    df['f0']=f0
    df['Q']=q
    df['Sign_inphase']=sign_inphase
    df['A0_Sign']=df['A0']*df['Sign_inphase']
    # Add single freq amplitude and phase into dataframe
    df['Amp_Single_Freq']=amp_single_freq
    df['Phase_Single_Freq']=phase_single_freq

    df['Amp_Array']=amp_array
    df['Freq_Array']=freq_array

    # Clean the table
    df2=df[['Tip_Bias','A0','f0','Q','Sign_inphase','A0_Sign','Amp_Single_Freq','Phase_Single_Freq','Amp_Array','Freq_Array']]

    # Seperate the write bias and read bias
    df3=df2.iloc[1::2]
    df3.reset_index(inplace=True)
    write_bias=df['Tip_Bias'][::2]
    df3['Write_Bias']=list(write_bias)
    df3.rename(columns={'Tip_Bias' : 'Read_Bias'},inplace=True)
    df4=df3[['Write_Bias','Read_Bias','A0','f0','Q','Sign_inphase','A0_Sign','Amp_Single_Freq','Phase_Single_Freq','Amp_Array','Freq_Array']]

    return df4

def main(data_dir,output_dir):

    os.chdir(data_dir)
    for f in os.listdir(data_dir):
        df=data_extract(f)
        df=fit_and_clean_table(df,100)
        filename=os.path.splitext(f)[0]
        filename=filename+'.csv'
        output_file=os.path.join(output_dir,filename)
        df.to_csv(output_file)
        print('Output {}'.format(filename))

data_dir=r'\\sbodata\Xfer\Xiaomin Chen\AFMi Test\SSPFM\2D Map on PZT\SSPFM Training\AutoCapture\Captured_After_20210412172903'
output_dir=r'C:\Users\Xiaomin.Chen\Desktop\AFMi-Test\Projects\SS-PFM\2D Map on PZT\Data1\CSV'

# data_dir=r'C:\Users\Xiaomin.Chen\Desktop\AFMi-Test\Projects\SS-PFM\2D Map on PZT\Data2\Test\Test dirIn'
# output_dir=r'C:\Users\Xiaomin.Chen\Desktop\AFMi-Test\Projects\SS-PFM\2D Map on PZT\Data2\Test\Test dirOut'

main(data_dir,output_dir)
print('Data Extraction Is Completed')






