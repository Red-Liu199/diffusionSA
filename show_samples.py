import json
import numpy as np
char2idx = json.load(open('data/text8/vocab.json', 'r'))
idx2char = {}
for c, i in char2idx.items():
    idx2char[i] = c
def decode(sentence: list):
    new_s = []
    for idx in sentence:
        new_s.append(idx2char[idx])
    return ''.join(new_s)
data = json.load(open('exp/DSA_cache/checkpoint/checkpoint_19_sampling.json', 'r'))
proposals, predictions, samples = np.array(data['proposals']), np.array(data['predictions']), np.array(data['samples']) # T+1, B, L
sent_idx = 1
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
    print(f'Timestep:{t}\n', decode(list(sentence)))
