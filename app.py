
import os
from flask import Flask, flash, redirect, render_template, request
import tensorflow as tf
import tensorflow_io as tfio
import subprocess
import numpy as np
import pandas as pd

UPLOAD_FOLDER = 'files'
app =  Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# This method manages the ajax post containing the .mp3 files from the js frontend
@app.route('/save-record', methods=['POST'])
def save_record():

    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    word_index = int(request.form['word']) - 1
    word_arr = ['SWING', 'UMBRELLA', 'QUEEN', 'HELICOPTER', 'ORANGE']
    file_name = word_arr[word_index]

    file_name_mp3 = str(file_name) + ".mp3"
    full_file_name_mp3 = os.path.join(app.config['UPLOAD_FOLDER'], file_name_mp3)
    
    # If the file exists, replace it, else just save
    if os.path.exists(full_file_name_mp3):
        os.remove(full_file_name_mp3)
    else:
        print("The file does not exist") 
    file.save(full_file_name_mp3)

    # Read in mp3 and convert to wav using ffmpeg
    file_name_wav = str(file_name) + ".wav"
    full_file_name_wav = os.path.join(app.config['UPLOAD_FOLDER'], file_name_wav)
    if os.path.exists(full_file_name_wav):
        os.remove(full_file_name_wav)
    else:
        print("The file does not exist") 

    subprocess.call(['ffmpeg', '-i', full_file_name_mp3,
                   full_file_name_wav])

    return '<h1>Success</h1>'


# This method loads in each of the models and runs each of the recorded words through them, it is acitvated when the website is posted to /submit (output.html)
@app.route("/submit", methods=['POST'])
def predict():
    
    word_list = ["HELICOPTER","ORANGE","QUEEN","SWING","UMBRELLA"]
    results_list = []
    
    # The below iterates over each model and appends predictions to the results_list
    # It is commented out in favour of a dummy results list as it can take up to 10 minutes to predict
    
    # for each in word_list:
    #     print(each)
    #     model = tf.keras.models.load_model('static/scripts/models/'+each)

    #     input_spectrogram = get_spectrogram('files/'+each+'.wav')
    #     input_spectrogram = np.array([input_spectrogram])
    #     stand_spectrogram = get_spectrogram('files/stand/stand_'+each+'.wav')
    #     stand_spectrogram = np.array([stand_spectrogram])
    #     y_hat = model.predict([input_spectrogram, stand_spectrogram])
    #     print(y_hat[0][0])
    #     results_list.append(y_hat[0][0])
    
    results_list = [0.01, 0.99, 0.78, 0.98, 0.99]
    
    # Multiply results to be percentages

    helicopter = int(results_list[0]*100)
    orange = int(results_list[1]*100)
    queen = int(results_list[2]*100)
    swing = int(results_list[3]*100)
    umbrella = int(results_list[4]*100)

    #Get data for diagnositc report
    headings = ("Sound Disorder", "May be present in word", "Summary", "Age to be resolved by")
    count_head = ("Sound Disorder", "Count of Potential Occurences")
    ordered_bespoke_df, disorder_count = output(results_list, word_list)

    return render_template('output.html', helicopter=helicopter, orange=orange, queen=queen, swing=swing, umbrella=umbrella, count_head=count_head, disorder_count=disorder_count, headings=headings, ordered_bespoke_df=ordered_bespoke_df)

# This method takes the results from the model and returns potential speech disorders that correlate with the disordered speech provided
def output(results_list, word_list):
    
    # Data containing possible SSD's, the words they can be present in, an explanation of what they may present as and an age when they should be resolved
    data = [
    ["Consonant Cluster Reduction", ["QUEEN", "SWING", "UMBRELLA"], ["Queen -> Keen/Ween", "Swing -> Sing/Wing", "Umbrella -> Umbella/Umrella"], "4.5 Years"],
    ["Weak Syllable Deletion", ["HELICOPTER", "UMBRELLA"], ["Less than correct number of syllables used", "Less than correct number of syllables used"], "4.5 Years"],
    ["Voicing", ["HELICOPTER"], ["Helicopter -> Heligopter"], "Atypical at all ages"],
    ["Backing", ["HELICOPTER"], ["Helicopter -> Helicopka"], "Atypical at all ages"],
    ["De-voicing", ["UMBRELLA"], ["Umbrella -> Umprella"], "Atypical at all ages"],
    ["Gliding", ["HELICOPTER", "UMBRELLA", "ORANGE"], ["Helicopter -> Heyecopter", "Umbrella -> Umbweja", "Orange -> Owange"], "6.5 Years"],
    ["Final Consonant", ["ORANGE", "QUEEN", "SWING"],["Orange -> Oran/Orin", "Queen -> Qwe", "Swing -> Swin"], "3 Years"],
    ["Fronting", ["HELICOPTER", "QUEEN", "SWING"],["Helicopter -> Helitopter", "Queen -> Tween", "Swing -> Swind"], "4.5 Years"],
    ["Stopping", ["SWING"], ["Swing -> Dwing/Twing"], "4.5 Years"],
    ["De-africation", ["ORANGE"], ["Orange -> Orind ('dj' sound to a 'd' sound)"], "4.5 Years"]
    ]

    returns = []
    df = pd.DataFrame(data=data)


    # The below iterates over the data, creating a new dataframe containing data bespoke to the diagnosis
    words_dis_index = [i for i,v  in enumerate(results_list) if v < 0.5]
    disordered_words = [word_list[i] for i in words_dis_index]

    for index, row in df.iterrows():
        for each in disordered_words:
            if each in row[1]:
                returns.append([row[0], each, row[2][row[1].index(each)], row[3]])



    ordered_bespoke_df = pd.DataFrame(returns)
    ordered_bespoke_df['count'] = ordered_bespoke_df.groupby(0)[0].transform('count')
    ordered_bespoke_df= ordered_bespoke_df.sort_values(by='count', ascending=False)
    ordered_bespoke_df = ordered_bespoke_df.drop('count', axis=1)

    disorder_count = pd.DataFrame(ordered_bespoke_df[0].value_counts())
    ordered_bespoke_df = list(ordered_bespoke_df.itertuples(index=False, name=None))
    disorder_count = list(disorder_count.itertuples(index=True, name=None))


    return ordered_bespoke_df, disorder_count
    

# These methods are direct from the model training so that the input speech has the same preprocessing as the training data
def load_wav_16k_mono(filename):
  # Load encoded wav file
  file_contents = tf.io.read_file(filename)
  # Decode wav (Tensors by channels)
  wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
  print(sample_rate, tf.size(wav))
  # Removing trailing axis?
  wav = tf.squeeze(wav, axis=-1)
  sample_rate = tf.cast(sample_rate, dtype=tf.int64)
  # Goes from 44100Hz to 16000Hz - amplitude of audio signal
  wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
  return wav

def get_spectrogram(filepath):
  #Get full wav
  wav = load_wav_16k_mono(filepath)
  #anchor_end_index = anchor_end * tf.cast(anchor_rate, dtype=('float64'))
  #Get the first 1.5 second of wav
  wav = wav[:24000]
  #if wav shorter than 1 second, pad with zeros

  zero_padding = tf.zeros([24000]-tf.shape(wav), dtype=tf.float32)
  wav = tf.concat([zero_padding, wav], 0)

  #Get spectrogram
  spectrogram = tf.signal.stft(wav, frame_length=480, frame_step=100)                
  spectrogram = tf.abs(spectrogram)
  spectrogram = tf.expand_dims(spectrogram, axis=2)
  
  return spectrogram

def predictor(input, standard):
    input_path = input + ".wav"
    standard_path = standard + ".wav"
    input_spec = get_spectrogram(input_path)
    standard_spec = get_spectrogram(standard_path)
    siamese_Model = tf.keras.models.load_model('static/scripts/models/'+input)
    y_hat = siamese_Model.predict([input_spec, standard_spec])
    print(y_hat)
    return y_hat; 

# This method takes the user from the instruction page (index.html) to the input page (input.html)
@app.route("/input", methods=['POST'])
def input():
    forward_message = "Moving Forward..."
    return render_template('input.html', forward_message=forward_message)

@app.route("/")
def hello():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
