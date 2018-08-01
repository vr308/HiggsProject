# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:44:45 2016

@author: vr308
"""

import matplotlib.pylab as plt
import numpy as np

s = 50
b = np.arange(1,1000,1)
y = y = np.sqrt(2*((s+b)*np.log(1+(s/b)) - s))
ys = (s+b)*np.log(1+(s/b))
y2= s/np.sqrt(b)

text = r'$(s+b)\ln{(1+ \frac{s}{b})}$'
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(b,ys)
plt.ylim(20,110)
plt.hlines(y=50,xmin=0,xmax=1000,color='r',linestyle='--')
plt.xlim(0,1000)
plt.grid(which='both')
plt.minorticks_on()
plt.xlabel('b : background count')
plt.ylabel(r'$(s+b)\ln{(1+ \frac{s}{b})}$')
plt.title("Asymptotic Expansion of " + text + " for s = 50 and " + r'$b \rightarrow \infty$',fontsize='small')

plt.subplot(122)
plt.plot(b,y2,label=r'$\frac{s}{\sqrt{b}}$')
plt.plot(b,y,label=r'$\sqrt{2((s+b)\ln({1+ \frac{s}{b}}) - s)}$')
plt.legend()
plt.title('AMS Objective for a fixed s = 50',fontsize='small')
plt.xlabel('b : background count')
plt.grid(which='both')
plt.minorticks_on()
plt.tight_layout()