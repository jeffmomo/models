import numpy as np
import pickle
import sys
import json

with open('/Scratch/dm116/annotations/test2017.json', 'r') as f:
  annotations = json.load(f)
  filename_id_map = {}
  category_id_map = {}
  for line in annotations['images']:
    filename_id_map[line['file_name']] = line['id']
  for line in annotations['categories']:
    category_id_map[line['name'].replace('×', '')] = int(line['id'])

with open('/Scratch/dm116/annotations/labels.txt', 'r') as f:
  idx_category_map = {}
  for line in f:
    idx, name = line.split(':')
    idx_category_map[int(idx)] = name.strip()

def translate_idx(idx):
  return category_id_map[idx_category_map[int(idx)]]

def get_filename(path):
  return path.split('/')[-1]

def get_filename_id(byte_filename):
  return filename_id_map[get_filename(str(byte_filename, 'utf-8'))]

c1, idx1, fname1 = pickle.load(open('out.pkl', 'rb'))
c2, idx2, fname2 = pickle.load(open('single_out.pkl', 'rb'))
c3, _, fname3 = pickle.load(open('single_out_2.pkl' , 'rb'))
c4, _, fname4 = pickle.load(open('single_out3.pkl', 'rb'))


visited = set()

print(idx1[:20], idx2[:20])
print(len(c1), len(c2))
incoming_filenames = [get_filename(str(x, 'utf8')) for x in fname2[:10]]
print('filenames', incoming_filenames)
print('filenames as ids', [filename_id_map[x] for x in incoming_filenames])
print('translated', [translate_idx(x) for x in idx1[:20]])

# print(list(set(map(get_filename, fname1)) - set(map(get_filename, fname2))))
print('lengths', 'cat_id', len(category_id_map), 'idx_cat', len(idx_category_map))

m = {}
idxm = {}

total = 0
correct = 0

for i in range(0, len(c1)):
  m[get_filename_id(fname1[i])] = c1[i] * 0.73
  # for label # idxm[fname1[i]] = idx1[i]
for i in range(0, len(c1)):
  m[get_filename_id(fname2[i])] += c2[i] * 0.70 #m[fname2[i]] + c2[i]
  m[get_filename_id(fname3[i])] += c3[i] * 0.71
  # m[get_filename_id(fname4[i])] += c4[i] * 0.69

# for i in range(0, len(c1)):
  #if np.argmax(m[fname1[i]]) == idxm[fname1[i]]:
  #  correct += 1
  #total += 1
  #total += 1
  #if np.argmax(c2[i]) == idx2[i]:
  #  correct += 1
  # print(np.argmax(c1[i])

# print(correct / total)

out_file = open('outfile.csv', 'w')
out_file.write('id,predicted\n')

#for i in range(0, len(c2)):
for fid, v in m.items():
  name = str(fid) #str(get_filename_id(v))
  if name not in visited:
    visited.add(name)
    # out_file.write(name + ',' + ' '.join([str(translate_idx(x)) for x in np.argpartition(c2[i], -5)[-5:]]))#c2[i].argsort()[::-1][:5]]))
    out_file.write(name + ',' + ' '.join([str(translate_idx(x)) for x in np.argpartition(v, -5)[-5:]]))#c2[i].argsort()[::-1][:5]]))
    out_file.write('\n')

out_file.close()
