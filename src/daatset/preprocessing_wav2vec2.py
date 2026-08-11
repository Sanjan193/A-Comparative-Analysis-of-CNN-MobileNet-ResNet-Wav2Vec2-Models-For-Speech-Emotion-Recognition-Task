'''Mel spectogram is Not used here'''
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

# this function is used to convert the voice datas in to 16kHz sampling rate
def audio_raw(audio_path):

  y,sr = lb.load(audio_path,sr=16000)
  max_len = 64000
  # keeping the max audion 4sec
  if len(y) < max_len:
    y = np.pad(
        y,
        (0,( max_len - len(y))),
        mode='constant'
    )
  # Truncate if too long
  else:
    y = y[:max_len]
  return y

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
         data = audio_raw(audio_path_p)
         normalized_audio = ((data - data.mean())/(data.std())) #using Z sore normalization here


        #  calling the function to store the arrays
         data_variable_1.append(normalized_audio)
         data_variable_2.append((emotion_index)-1) #doing -1 because the out put of model will be in range 0-7 so fixing the max label upto 8 would throw an eror while calcularting crossentropy loss





