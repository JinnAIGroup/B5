"""   JLL, 2022.3.15
(YPN) jinn@Liu:~/YPN/B5$ python train_plot.py
"""
import numpy as np
import matplotlib.pyplot as plt

plt.subplot(311)
train_loss2 = np.loadtxt('./saved_model/train_loss_out2.txt')
t2, = plt.plot(train_loss2, 'k--', label='train2')
valid_loss2 = np.loadtxt('./saved_model/valid_loss_out2.txt')
v2, = plt.plot(valid_loss2, 'g--', label='valid2')
plt.ylabel("loss")
legend1 = plt.legend([t2], ['train2'], edgecolor='None', loc='upper right')
plt.legend([v2], ['valid2'], edgecolor='None', loc='lower left')
plt.gca().add_artist(legend1)

plt.subplot(312)
train_rmse2 = np.loadtxt('./saved_model/train_rmse_out2.txt')
t2, = plt.plot(train_rmse2, 'k--', label='train2')
valid_rmse2 = np.loadtxt('./saved_model/valid_rmse_out2.txt')
v2, = plt.plot(valid_rmse2, 'g--', label='valid2')
plt.ylabel("rmse")
legend1 = plt.legend([t2], ['train2'], edgecolor='None', loc='upper right')
plt.legend([v2], ['valid2'], edgecolor='None', loc='lower left')
plt.gca().add_artist(legend1)

plt.subplot(313)
train_mae2 = np.loadtxt('./saved_model/train_mae_out2.txt')
t2, = plt.plot(train_mae2, 'k--', label='train2')
valid_mae2 = np.loadtxt('./saved_model/valid_mae_out2.txt')
v2, = plt.plot(valid_mae2, 'g--', label='valid2')
plt.ylabel("mae")
plt.xlabel("epoch")
legend1 = plt.legend([t2], ['train2'], edgecolor='None', loc='upper right')
plt.legend([v2], ['valid2'], edgecolor='None', loc='lower left')
plt.gca().add_artist(legend1)

plt.draw()
plt.savefig('./saved_model/modelB5_outs.png')
plt.pause(0.5)
input("Press ENTER to close ...")
plt.close()
