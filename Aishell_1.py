import os
import json
import random
import soundfile as sf


def load_data(speech_path, bk_path):
    speech_wav = {}
    bk_list = []
    for bk in os.listdir(bk_path):
        bk_list.append(bk_path + '/{}'.format(bk))
    for person in os.listdir(speech_path):
        wav_list = []
        for wav in os.listdir(speech_path + '/{}'.format(person)):
            wav_list.append(speech_path + '/{}/{}'.format(person,wav))
        speech_wav[person] = wav_list
    return speech_wav, bk_list


def add_noise(speech_wav_dic, bk_wav_list, wav_save_path, setname):
    wav_dic = {}
    for speaker in speech_wav_dic.keys():
        wav_list = []
        bk_ground_sig,sr = sf.read(random.choice(bk_wav_list))
        exists = os.path.isdir(os.path.join(wav_save_path, speaker))
        if exists:
            print("wav saved in {}".format(wav_save_path + speaker))
        else:
            print("build {}".format(os.path.join(wav_save_path, speaker)))
            os.makedirs(os.path.join(wav_save_path, speaker))
        print('processing speaker {}'.format(speaker))
        for wav in speech_wav_dic[speaker]:
            wav_name = os.path.basename(wav)
            wav_list.append(wav_name)
            sig,sr = sf.read(wav)
            length = len(sig)
            end = len(bk_ground_sig)
            start = random.randint(0, end - length)
            new_wav = bk_ground_sig[start: start+length] + sig
            sf.write(os.path.join(wav_save_path, speaker, wav_name),new_wav,sr)
        wav_dic[speaker]= wav_list
    json_file = '{}/{}.json'.format(os.path.dirname(wav_save_path), setname)
    with open(json_file, 'w') as f:
        json.dump(wav_dic, f)
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
    for i in ['trainset', 'valset', 'testset']:
        speech_path = 'dataset/{}_clean'.format(i)
        speech_wav, bk_list = load_data(speech_path, bk_path)
        #
        wav_save_path = 'dataset/{}_noisy'.format(i)
        json_file = add_noise(speech_wav, bk_list, wav_save_path, i)
        #
        wav_list = read_json(json_file)
        len(wav_list)


if __name__ == "__main__":
    main()