import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
step = []

max_len = 201
num_run = 3

ours_val_ndcg = np.zeros(max_len)
ours_val_hr = np.zeros(max_len)
ours_test_ndcg = np.zeros(max_len)
ours_test_hr = np.zeros(max_len)

for i in range(num_run):
    with open('/home/hjhwang/Codes/SASRec/ml-1m_default/log' + str(i+1) + '.txt', 'r') as f:
        lines = f.read().splitlines()
        # last_line = lines[-1]
        # print(last_line)

    for i in range(max_len):
        line = lines[i].split()
        ours_val_ndcg[i] += float(line[0][1:-1])
        ours_val_hr[i] += float(line[1][:-1])
        ours_test_ndcg[i] += float(line[2][1:-1])
        ours_test_hr[i] += float(line[3][:-1])

ours_val_ndcg /= 3.0
ours_val_hr /= 3.0
ours_test_ndcg /= 3.0
ours_test_hr /= 3.0

SASRec_val_ndcg = np.zeros(max_len)
SASRec_val_hr = np.zeros(max_len)
SASRec_test_ndcg = np.zeros(max_len)
SASRec_test_hr = np.zeros(max_len)

for i in range(1):
    with open('/home/hjhwang/Codes/SASRec/ml-1m_default/SASRec_log' + str(i+1) + '.txt', 'r') as f:
        lines = f.read().splitlines()
        # last_line = lines[-1]
        # print(last_line)

    for i in range(max_len):
        line = lines[i].split()
        SASRec_val_ndcg[i] += float(line[0][1:-1])
        SASRec_val_hr[i] += float(line[1][:-1])
        SASRec_test_ndcg[i] += float(line[2][1:-1])
        SASRec_test_hr[i] += float(line[3][:-1])

SASRec_val_ndcg /= 1.0#3.0
SASRec_val_hr /= 1.0#3.0
SASRec_test_ndcg /= 1.0#3.0
SASRec_test_hr /= 1.0#3.0


# For test set
plt.plot(ours_test_ndcg, label='Ours')
plt.plot(SASRec_test_ndcg, label='SASRec')
plt.xlabel("Epochs")
plt.ylabel("NDCG")
plt.legend()
plt.show()

plt.plot(ours_test_hr, label='Ours')
plt.plot(SASRec_test_hr, label='SASRec')
plt.xlabel("Epochs")
plt.ylabel("Hit Rate")
plt.legend()
plt.show()

# For valid set
plt.plot(ours_val_ndcg, label='Ours')
plt.plot(SASRec_val_ndcg, label='SASRec')
plt.xlabel("Epochs")
plt.ylabel("NDCG")
plt.legend()
plt.show()

plt.plot(ours_val_hr, label='Ours')
plt.plot(SASRec_val_hr, label='SASRec')
plt.xlabel("Epochs")
plt.ylabel("Hit Rate")
plt.legend()
plt.show()