import matplotlib.pyplot as plt
import numpy as np
T=5
s_list = [0, 0.008, 0.2, 0.8]
plt.figure()
for s in s_list:
    f_0 = np.cos(s*np.pi/(2*(s+1)))
    t = np.linspace(0, T, T+1)
    f_t = np.cos((t/T+s)*np.pi/(2*(1+s)))
    plt.scatter(t, f_t/f_0)
    # plt.plot(t, f_t)
plt.legend(s_list)
plt.savefig('exp/figure.png')