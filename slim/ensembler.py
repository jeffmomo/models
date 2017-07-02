import numpy as np
import pickle
import sys

c1, idx1, fname1 = pickle.load(open('out.pkl', 'rb'))
c2, idx2, fname2 = pickle.load(open('single_out.pkl', 'rb'))
print(idx1[:20], idx2[:20])
print(len(c1), len(c2))
print(fname2[:10])
sys.exit(0)


m = {}
idxm = {}

total = 0
correct = 0
for i in range(0, len(c1)):
  m[fname1[i]] = c1[i]
  idxm[fname1[i]] = idx1[i]
for i in range(0, len(c1)):
  m[fname2[i]] = c2[i] #m[fname2[i]] + c2[i]

for i in range(0, len(c1)):
  if np.argmax(m[fname1[i]]) == idxm[fname1[i]]:
    correct += 1
  total += 1
  #total += 1
  #if np.argmax(c2[i]) == idx2[i]:
  #  correct += 1
  # print(np.argmax(c1[i])

print(correct / total)
