import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import os
import pyaudio
import wave
import uuid
from tqdm import tqdm
from pydub import AudioSegment

# Load the pre-trained model
model = tf.keras.models.load_model('models/Best_DenseNet121_model.h5')

# Define class labels with corresponding bird species names
class_labels = ['Greylag Goose', 'Whooper Swan', 'Mallard Duck', 'Green-winged Teal', 'Grey Heron',
                'Cormorant', 'Black-winged Stilt', 'Lapwing', 'Wood Sandpiper', 'Eurasian Tree Sparrow']

# Define dictionary mapping labels to bird species image paths and descriptions
bird_data = {
    'Greylag Goose': {'image_path': 'images/grey_goose.jpg',
                   'description': '体重1900-2350克，体长770-795毫米。雌雄两性全年体色概为灰褐色，嘴基周围有狭长的白纹。'
                                  '下背及腰部为较深灰鼠色、颈、胸和腹部淡灰色，腹部有黑色横纹眼棕色，嘴橙黄或略带红色。'
                                  '在野外观察中，本物种最大的特征就是没有特征，它不具有豆雁喙端醒目的黄色、鸿雁颜色对比鲜明的颈部，'
                                  '也不具有白额雁和小白额雁引人瞩目的白色额头，因此如果在野外看到一只灰褐色的大型雁类又没有观察到上述特征，'
                                  '且可以肯定地观察到粉红色的喙和脚，那么就可以判断其为灰雁。'},
    'Whooper Swan': {'image_path': 'images/swan.jpg',
             'description': '大天鹅是体形较大的雁形目鸟类，本种体形较与之混群的小天鹅略大，但只有两者同时出现在观察者的视野中时才能明确地观察到这种差异，'
                            '雄体长1.5米以上，雌体较小，雄性体重为7.4-14千克，平均为9.8千克，雌性则仅8.2-9.2千克。大天鹅雄雌同形同色，通体洁白，颈部极长；'
                            '体态优雅。它们的脚和腿都是黑色的。喙基部的一半呈橙黄色，尖端呈黑色。在春季和夏季，由于富含铁的环境，成年大天鹅可能会出现深色的颈部羽毛。'
                            '喙上黄黑两种颜色形成的团因个体而异，可以用于个体识别。雏鸟通常是白色的，但也有些是灰色的。'
                            '本物种显著的鉴别特征在喙部，大天鹅的喙部由黑黄两色组成，黄色区域位于喙的基部，与小天鹅相比大天鹅喙部的黄色区域更大，超过了鼻孔的位置。'},
    'Mallard Duck': {'image_path': 'images/anas_platyrhynchos.jpg',
                     'description': '绿头鸭（学名：Anas platyrhynchos）又名大头绿（雄）、蒲鸭（雌）'
                                    '、野鸭，古称鹜[注 1]，家鸭是其驯化亚种。绿头鸭飞行速度可达到每小时65公里。'},
    'Green-winged Teal': {'image_path': 'images/anas_crecca.jpg',
                          'description': '绿翅鸭是体形最小的河鸭属鸟类之一，体长不到40公分，仅为常见的绿头鸭的2/3。本物种雄雌异形异色，“雄鸟”头颈部基色为栗褐色，从两侧眼周开始直到颈侧分布者一条绿色的色带，呈逗号的形状，与栗褐色的底色形成鲜明的对比，而脸上的这个“绿色大逗号”也是辨识绿翅鸭雄鸟的重要特征，除了大逗号，从嘴基开始有一条淡淡的白色细线延伸到眼前；上背、肩部、两胁远看灰色，进看则为白色底色上密布黑色幼细横纹；下背和腰部褐色，尾上覆羽黑色；翼镜为与头部“大逗号”相同的翠绿色，初级飞羽最外侧一枚白色，当双翅收拢时在上体和下体之间形成一条醒目的白色横带，这也是鉴别本物种的重要特征；胸部和上腹部淡褐色，具深褐色的圆斑，尾下覆羽为奶黄色，在臀部形成一块具有黑色绒边的奶黄色块，这是鉴别本物种的第三大特征。\'雌鸟\'为雄鸟的暗色版本，通体以褐色为基调，并不具有雄鸟所具有的“面部大逗号”、“体侧白线”和“奶油屁股”这三大特征，但雌鸟保持了翠绿色的翼镜，并有着非常小巧的身材，这都是本物种雌鸟的鉴别特征。虹膜褐色；嘴和脚都是灰色。　'},
    'Grey Heron': {'image_path': 'images/ardea_cinerea.jpg',
              'description': '苍鹭体长 84－102 厘米，翼展 155－195 厘米，高约 100 厘米，体重 1.02－2.08 公斤。其上半身主要为灰色，腹部为白色。成鸟的过眼纹'
                             '及冠羽黑色，飞羽、翼角及两道胸斑黑色，头、颈、胸及背白色，颈具黑色纵纹，余部灰色。幼鸟的头及颈灰色较重，但无黑色。'
                             '虹膜－黄色；喙－黄绿色；脚－偏黑。叫声：深沉的喉音呱呱声及似鹅的叫声。'},
    'Cormorant': {'image_path': 'images/cormorant.jpg',
                  'description': '普通鸬鹚：体长90厘米，有偏黑色闪光，嘴厚重，脸颊及喉白色。繁殖期颈及头饰以白色丝状羽，'
                                 '两胁具白色斑块。幼鸟：深褐色，下体污白。 虹膜为蓝色；喙为黑色，下嘴基裸露皮肤黄色；脚为黑色。'},
    'Black-winged Stilt': {'image_path': 'images/himantopus.jpg',
                           'description': '黑翅长脚鹬体态修长，体长越35厘米，通体黑白分明，一双红腿很容易辨认。'
                                          '黑翅长脚鹬的喙黑色细长，雄鸟繁殖羽的从眼后到头顶和颈后黑色，背部和翅为黑色，与颈后的黑色不相连，'
                                          '其他部位为白色；雌鸟的繁殖羽在头和颈后没有黑色，眼后有灰色斑。亚成体的鸟色块分布接近成鸟，但颜色'
                                          '较浅，为灰褐色。黑翅长脚鹬的腿和跗趾修长，为红色，亚成体鸟的腿颜色略浅，显橘红色；飞行时长腿拖于'
                                          '尾后，是重要的辨识特征。虹膜粉红色。'},
    'Lapwing': {'image_path': 'images/vanellus.jpg',
              'description': '风头麦鸡：一种鸻科麦鸡属的鸟类，多数是候鸟。每年夏天在中欧、东欧、哈萨克至中国东北一带繁殖，'
                             '冬天到华南、台湾、日本、印度、西亚、法国、伊比利半岛和北非越冬。'},
    'Wood Sandpiper': {'image_path': 'images/woodcock.jpg',
                 'description': '林鹬（学名：Tringa glareola）又名鹰斑鹬，为鹬科鹬属的鸟类。在中国大陆，'
                                '分布于华北和香港等地。该物种的模式产地在瑞典。'},
    'Eurasian Tree Sparrow': {'image_path': 'images/sparrow.jpg',
                'description': '麻雀体形较小，体长为12.5至14公分左右[2]，体形短圆，具有典型的食谷鸟特征。麻雀雄雌同形同色，头顶和后颈为栗色，'
                               '面部白色，双颊中央各自有一块黑色色块，这块黑色的小脸蛋是鉴别麻雀的关键特征，有研究指出，在群体中地位越高的个体喉部的黑色区域也相应越大，'
                               '黑色也越饱满；上体褐色，具黑色斑点，所有飞羽、小覆羽、初级覆羽均为黑褐色；具两道污白色翅斑；尾羽褐色；下体污白色；虹膜为深褐色；'
                               '喙呈圆锥形，比较粗壮，呈黑色；足为粉褐色。（见图）'}
}


# Adjust the spectrogram shape
def adjust_spectrogram_shape(ps):
    target_shape = (1, 128, 128, 1)
    current_shape = ps.shape

    if current_shape == target_shape:
        return ps

    if current_shape[2] < target_shape[2]:
        ps = np.pad(ps, ((0, 0), (0, 0), (0, target_shape[2] - current_shape[2]), (0, 0)), mode='constant')
    elif current_shape[2] > target_shape[2]:
        ps = ps[:, :, :target_shape[2], :]

    return ps


# Load audio data, preprocess, and denoise
def load_data(audio_file):
    wav, sr = librosa.load(audio_file, sr=16000)

    # Preemphasis
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])

    # Denoise audio
    noisy_part = wav[:8000]  # Assuming 8000 samples have noise
    noisy_reduced = librosa.effects.preemphasis(noisy_part)
    wav_output = np.concatenate(
        [noisy_reduced] + [wav[sliced[0]:sliced[1]] for sliced in librosa.effects.split(wav, top_db=20)])

    # Apply Hamming window
    n_fft = 2048
    frames = librosa.util.frame(wav_output, frame_length=n_fft, hop_length=512)
    frames = frames * np.hamming(n_fft)[:, np.newaxis]

    # Extract mel spectrogram
    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256, n_fft=n_fft).astype(np.float32)
    ps = ps[np.newaxis, ..., np.newaxis]

    # Adjust the spectrogram shape
    ps = adjust_spectrogram_shape(ps)

    return ps

# Function to predict bird species
def predict_bird_species(audio_data):
    prediction = model.predict(audio_data)
    label_index = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[label_index]
    return predicted_label


# Streamlit app
st.title("Bird Species Identification")

# Record audio
def record_audio(filename, duration=5, chunk=1024, channels=1, rate=16000):
    p = pyaudio.PyAudio()
    frames = []

    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    st.write(f"Recording... (max duration: {duration} seconds)")

    for _ in tqdm(range(0, int(rate / chunk * duration))):
        data = stream.read(chunk)
        frames.append(data)

    st.write("Recording completed!")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b"".join(frames))
    wf.close()

# Function to convert audio to wav format
def convert_to_wav(audio_file):
    wav_filename = os.path.join(os.getcwd(), str(uuid.uuid4()) + ".wav")  # Save in current directory
    sound = AudioSegment.from_file(audio_file)
    sound.export(wav_filename, format="wav")
    return wav_filename

# Main function
def main():
    st.sidebar.subheader("Options")
    audio_option = st.sidebar.selectbox("Select audio input option", ("Record Audio", "Upload an audio file"))

    if audio_option == "Record Audio":
        start_recording = st.sidebar.button("Start Recording")
        if start_recording:
            filename = os.path.join(os.getcwd(), str(uuid.uuid4()) + ".wav")  # Save in current directory
            record_audio(filename)
            audio_file = filename
        else:
            audio_file = None
    else:
        audio_file = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

    if audio_file:
        st.audio(audio_file, format="audio/wav")

        if isinstance(audio_file, str) or audio_file.name.endswith(".wav"):
            wav_file = audio_file
        else:
            st.write("Converting audio to WAV format...")
            wav_file = convert_to_wav(audio_file)
            st.write("Conversion completed!")

        st.write("Predicting bird species... Please wait.")
        audio_data = load_data(wav_file)
        predicted_species = predict_bird_species(audio_data)
        st.success("Prediction complete!")
        st.subheader("Bird Species Prediction:")
        st.write(f"The predicted bird species is: {predicted_species}")

        # Display bird image and description based on predicted species
        if predicted_species in bird_data:
            bird_info = bird_data[predicted_species]
            st.image(bird_info['image_path'], caption=f"{predicted_species} Image", use_column_width=True)
            st.write(f"Description: {bird_info['description']}")
        else:
            st.warning("No image available for the predicted bird species.")

if __name__ == "__main__":
    main()


