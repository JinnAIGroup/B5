"""   JLL, 2022.2.5
(YPN) jinn@Liu:~/YPN/B5$ python plot_txtB5.py
"""
import numpy as np
import matplotlib.pyplot as plt

plt.subplot(311)
train_loss1 = np.loadtxt('./saved_model/train_loss_out1.txt')
t1, = plt.plot(train_loss1, 'b', label='train1')
train_loss2 = np.loadtxt('./saved_model/train_loss_out2.txt')
t2, = plt.plot(train_loss2, 'k--', label='train2')
valid_loss1 = np.loadtxt('./saved_model/valid_loss_out1.txt')
v1, = plt.plot(valid_loss1, 'r', label='valid1')
valid_loss2 = np.loadtxt('./saved_model/valid_loss_out2.txt')
v2, = plt.plot(valid_loss2, 'g--', label='valid2')
#plt.title("Model Loss")
plt.ylabel("loss")
#plt.xlabel("epoch")
legend1 = plt.legend([t1, t2], ['train1', 'train2'], edgecolor='None', loc='upper right')
plt.legend([v1, v2], ['valid1', 'valid2'], edgecolor='None', loc='lower left')
plt.gca().add_artist(legend1)
#plt.legend(['train1', 'train2', 'valid1', 'valid2'], edgecolor='None', loc='upper right', ncol=2, handleheight=1.0, labelspacing=0.05)

plt.subplot(312)
train_rmse1 = np.loadtxt('./saved_model/train_rmse_out1.txt')
t1, = plt.plot(train_rmse1, 'b', label='train1')
train_rmse2 = np.loadtxt('./saved_model/train_rmse_out2.txt')
t2, = plt.plot(train_rmse2, 'k--', label='train2')
valid_rmse1 = np.loadtxt('./saved_model/valid_rmse_out1.txt')
v1, = plt.plot(valid_rmse1, 'r', label='valid1')
valid_rmse2 = np.loadtxt('./saved_model/valid_rmse_out2.txt')
v2, = plt.plot(valid_rmse2, 'g--', label='valid2')
plt.ylabel("rmse")
legend1 = plt.legend([t1, t2], ['train1', 'train2'], edgecolor='None', loc='upper right')
plt.legend([v1, v2], ['valid1', 'valid2'], edgecolor='None', loc='lower left')
plt.gca().add_artist(legend1)

plt.subplot(313)
train_mae1 = np.loadtxt('./saved_model/train_mae_out1.txt')
t1, = plt.plot(train_mae1, 'b', label='train1')
train_mae2 = np.loadtxt('./saved_model/train_mae_out2.txt')
t2, = plt.plot(train_mae2, 'k--', label='train2')
valid_mae1 = np.loadtxt('./saved_model/valid_mae_out1.txt')
v1, = plt.plot(valid_mae1, 'r', label='valid1')
valid_mae2 = np.loadtxt('./saved_model/valid_mae_out2.txt')
v2, = plt.plot(valid_mae2, 'g--', label='valid2')
plt.ylabel("mae")
plt.xlabel("epoch")
legend1 = plt.legend([t1, t2], ['train1', 'train2'], edgecolor='None', loc='upper right')
plt.legend([v1, v2], ['valid1', 'valid2'], edgecolor='None', loc='lower left')
plt.gca().add_artist(legend1)

plt.draw()
plt.savefig('./saved_model/modelB5_outs.png')
plt.pause(0.5)
input("Press ENTER to close ...")
plt.close()
