import os
import json
import numpy as np
import argparse
import sentencepiece as spm
import random
def main(args):
    cfg = json.load(open(args.config_path, 'r'))
    if cfg['others']['character_level']:
        char2idx = json.load(open('data/text8/vocab.json', 'r'))
        idx2char = {}
        for c, i in char2idx.items():
            idx2char[i] = c
        def decode_func(sentence: list):
            new_s = []
            for idx in sentence:
                new_s.append(idx2char[idx])
            return ''.join(new_s)
    else:
        vocab_size = cfg['others']['vocab_size']
        tokenizer = spm.SentencePieceProcessor(model_file=f'data/text8/spm_model{vocab_size}.model')
        decode_func = tokenizer.decode
    data = json.load(open(args.samples_path, 'r'))
    proposals, predictions, samples = np.array(data['proposals']), np.array(data['predictions']), np.array(data['samples']) # T+1, B, L
    print(proposals.shape)
    sent_idx = random.randint(0, proposals.shape[2]-1)
    one_series_proposal_sample = proposals[:, :, sent_idx, :] # T+1, L
    one_series_prediction_sample = predictions[:, :, sent_idx, :]
    one_series_sample = samples[:, :, sent_idx, :]
    # print(one_series_prediction_sample.shape)
    # print('Proposals:')
    # for t, sentence in enumerate(one_series_proposal_sample[0]):
    #     print(f'Timestep:{t}\n', decode(list(sentence)))
    # print('Predictions:')
    # for t, sentence in enumerate(one_series_prediction_sample[0]):
    #     print(f'Timestep:{t}\n', decode(list(sentence)))
    print('Samples:')
    for t, sentence in enumerate(one_series_sample[0]):
        print(f'Timestep:{t}\n', decode_func(sentence.tolist()))
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # testing
    parser.add_argument('--samples_path', type=str)
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()
    main(args)
