import os
import json
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm


def load_data(speech_path, noise_path):
    speech_list = []
    noise_list = []
    for bk in tqdm(os.listdir(noise_path)):
        noise_list.append(noise_path + '/{}'.format(bk))
    for person in tqdm(os.listdir(speech_path)):
        wav_list = []
        for wav in os.listdir(speech_path + '/{}'.format(person)):
            wav_list.append(speech_path + '/{}/{}'.format(person, wav))
        speech_list.extend(wav_list)
    return speech_list, noise_list


def speech_join_noise(speech_list, noise_list):
    speech_size = len(speech_list)
    noise_size = len(noise_list)
    speech_noise_list = []
    for i in range(speech_size):
        speech_noise_dic = {}
        if i >= noise_size:
            j = i - noise_size*int(i/noise_size)
        else:
            j = i
        speech_noise_dic["clean_speech"] = speech_list[i]
        speech_noise_dic["noise"] = noise_list[j]
        speech_noise_list.append(speech_noise_dic)
    return speech_noise_list


def add_noise(speech_noise_list, setname):
    noisy_speech_list = []
    speech_dic = {}
    for i in tqdm(speech_noise_list):
        clean_speech = i["clean_speech"]
        speaker = clean_speech.split('/')[-2]
        speech_dic[speaker] = []
    for i in tqdm(speech_noise_list):
        clean_speech = i["clean_speech"]
        speaker = clean_speech.split('/')[-2]
        noise = i["noise"]
        wav_save_path = os.path.dirname(clean_speech).replace('clean', 'noisy')
        wav_name = os.path.basename(clean_speech)
        if not os.path.exists(wav_save_path):
            os.makedirs(wav_save_path)
        noisy_speech = os.path.join(wav_save_path, wav_name)
        sig, sr = sf.read(clean_speech)
        bk_ground_sig, sr = sf.read(noise)
        if len(sig) < 16000:
            continue
        #
        speech_dic[speaker].append(wav_name)
        noisy_speech_list.append(noisy_speech)
        #
        p_sig = np.sum(abs(sig) ** 2)
        if p_sig <= 10:
            SNR = round(np.random.uniform(20, 20), 2)
        elif 10 < p_sig <= 50:
            SNR = round(np.random.uniform(15, 20), 2)
        elif 50 < p_sig <= 100:
            SNR = round(np.random.uniform(10, 15), 2)
        elif 100 < p_sig <= 200:
            SNR = round(np.random.uniform(5, 10), 2)
        else:
            SNR = round(np.random.uniform(0, 5), 2)
        background_volume = p_sig / 10 ** (SNR / 10)
        # background signal
        length = len(sig)
        end = len(bk_ground_sig)
        if length > end:
            bk_ground_sig = np.tile(bk_ground_sig, int(np.ceil(length/end)))
        start = random.randint(0, max(0, end - length))
        background_buffer = bk_ground_sig[start: start + length]
        background_buffer = np.sqrt(background_volume / p_sig) * background_buffer
        # add noise
        new_wav = background_buffer + sig
        # saving
        sf.write(noisy_speech, new_wav, sr, subtype='PCM_32')

    json_file = '{}/{}.json'.format('dataset', setname)
    with open(json_file, 'w') as f:
        json.dump(speech_dic, f)
    return json_file


def read_json(json_path):
    wav_list = []
    with open(json_path, 'r') as f:
        wav_dic = json.loads(f.readlines()[0])
    for i in wav_dic.keys():
        for wav in wav_dic[i]:
            wav_file = '{}/{}'.format(i,wav)
            wav_list.append(wav_file)
    return wav_list


def main():
    bk_path = 'dataset/__background__'
    for i in ['trainset', 'valset', 'testset']:  #
        speech_path = 'dataset/{}_clean'.format(i)
        speech_list, bk_list = load_data(speech_path, bk_path)
        #
        speech_noise_list = speech_join_noise(speech_list, bk_list)

        json_file = add_noise(speech_noise_list, i)
        #
        wav_list = read_json(json_file)
        len(wav_list)


if __name__ == "__main__":
    main()
