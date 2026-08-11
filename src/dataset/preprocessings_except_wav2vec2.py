'''Mel spectogram is used for all the models except the Wav2Vec2 Model'''
# pasting the file path here

def signal_wave(dataset_path,actor_idx):

    count = 1
    actor = dataset_path +'/'+ actor_idx
    for sample in os.listdir(actor)[:10]:

       spf = wave.open(f"{dataset_path}" +"/" + f"{actor_idx}"+"/"+sample, "r") #one speech of actor-1 exmaple
       path = f"{dataset_path}" +"/" + f"{actor_idx}"+"/"+sample

       # Extracting Raw Audio from Wav File
       signal = spf.readframes(-1)
       signal = np.fromstring(signal, np.int16)

       # If Stereo
       if spf.getnchannels() == 2:
         print("Just mono files")
         sys.exit(0)
       plt.figure(1)
       plt.title(f"Signal Wave {count} for {actor_idx} ")
       plt.ylabel("Frequency/Amplitude")
       plt.xlabel("Time")
       plt.plot(signal)
       plt.show()
       count+=1
       display(IPython.display.Audio(path))
# plotting Mel-Spectograms
def mel_spec(dataset_path,actor_idx):

    count = 1
    actor = dataset_path +'/'+ actor_idx
    for sample in os.listdir(actor)[:10]:

       path = (f"{dataset_path}" +"/" + f"{actor_idx}"+"/"+sample) #one speech of actor-1 exmaple
       y,sr = lb.load(path,sr=16000)
       ml_specto = lb.feature.melspectrogram(y=y,sr=sr)


       log_mel_spectogram = lb.power_to_db(ml_specto,ref=np.max)
       plt.figure(figsize=(10,5))

       plt.title(f'Mel Spectrogram diagram {count} For {actor_idx}')
       plt.ylabel('Mel Bands(value corresponding to Mel scale)')
       plt.xlabel('Time Frames')
       lb.display.specshow(log_mel_spectogram, sr=sr, x_axis='time', y_axis='log');
       plt.colorbar(format='%+2.0f dB')
       plt.show()
       display(IPython.display.Audio(path))
       count += 1


# keeping the size of the spectrogram heatmap matrix in 128,128
def mel_specto(audio_path):

  y,sr = lb.load(audio_path)
  ml_specto = lb.feature.melspectrogram(y=y,sr=sr)
  # for better CNN(later I'll do) optimisationn keeping the each mfcc matrix shape for each audio example as (128,128)
  max_len = 128

  # doing all the column at same length because num py array needs all the column to be same shaped
  # Pad if too short
  if ml_specto.shape[1] < max_len:
    ml_specto = np.pad(
        ml_specto,
        ((0, 0), (0, max_len - ml_specto.shape[1])),
        mode='constant'
    )
  # Truncate if too long than 128
  else:
    ml_specto = ml_specto[:, :max_len]

  # returning the MFCC matrix ad numpy array
  # here the shape is (128, 143) so the 128 is the feature
  return ml_specto


# for storng the data in a variable
def for_storing_data(data_path,usel_less_dir,data_variable_1,data_variable_2):
 for actor in os.listdir(data_path):

    # skipping extra files
    if actor != usel_less_dir :

      for wave in (os.listdir(data_path+'/'+actor)):

         emotion_index = wave.split("-")[2]
         emotion_index =int(emotion_index)

        #  normalizing mel specto gram matrix with Z score normalizatin
         audio_path_p= data_path +'/'+actor+'/'+wave
         data = mel_specto(audio_path_p)
         normalized_audio = ((data - data.mean())/(data.std())) #using Z sore normalization here


        #  calling the function to store the arrays
         data_variable_1.append(normalized_audio)
         data_variable_2.append((emotion_index)-1) #doing -1 because the out put of model will be in range 0-7 so fixing the max label upto 8 would throw an eror while calcularting crossentropy loss
