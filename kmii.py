import numpy as np
import scipy.integrate as si
import vpython as v

class zvrk:
    def __init__(self,I1,I3,O1,O2,O3,theta0,psi0,t,steps=2000):
        self.I1 = I1
        self.I2 = I1
        self.I3 = I3
        self.O1 = O1
        self.O2 = O2
        self.O3 = O3
        self.theta0 = np.deg2rad(theta0)
        self.psi0 = np.deg2rad(psi0)
        self.tm = t
        self.steps = int(steps)

    def calc(self):
        self.w = self.O3*(self.I3-self.I1)/self.I1 #precesija

        self.E = 1/2 * ((self.I1*self.O1)**2 + (self.I2*self.O2)**2 + (self.I3*self.O3)**2)

        self.M3 = self.O3*self.I3

        self.dt = self.tm/self.steps

        self.listaO1 = [self.O1]
        self.listaO2 = [self.O2]
        self.listaO3 = [self.O3]
        self.listat = [0]
        self.t = 0
        for i in range(self.steps):
            self.O1 = self.O1*np.cos(self.w*self.dt) + self.O2*np.sin(self.w*self.dt)
            self.listaO1.append(self.O1)
            self.O2 = self.O2*np.cos(self.w*self.dt) - self.O1*np.sin(self.w*self.dt)
            self.listaO2.append(self.O2)
            self.O3 = self.O3
            self.listaO3.append(self.O3)
            self.t += self.dt
            self.listat.append(self.t)

    def int(self):
        def f(y,x):
            self.o1 = y[0]
            self.o2 = y[1]
            self.dto1 = -self.w * self.o2
            self.dto2 = self.w * self.o1
            return [self.dto1, self.dto2]
        self.y0 = [self.O1, self.O2]
        tx=np.linspace(0, 10, 1000)
        y=si.odeint(f,self.y0, tx) 

        self.o1 = y[:,0]
        self.o2 = y[:,1]

    
    def sim(self):
        prostor = v.canvas(title='sim', width=1000, height=500, background=v.vector(0.8,0.8,0.8))

        graph = v.graph(xtitle='t / s', ytitle='\u03A9 / rad s<sup>-1</sup>', align="left")
        graph1 = v.gcurve(color=v.color.cyan, label="\u03A9<sub>1</sub>")
        graph2 = v.gcurve(color=v.color.green, label="\u03A9<sub>2</sub>")
        graph3 = v.gcurve(color=v.color.red, label="\u03A9<sub>3</sub>")
        graph1a = v.gdots(color=v.color.orange, label = "\u03A9<sub>1a</sub>", interval=50)
        graph2a = v.gdots(color=v.color.red, label = "\u03A9<sub>2a</sub>",interval=50)
        graph3a = v.gdots(color=v.color.blue, label = "\u03A9<sub>3a</sub>",interval=50)

        for i in range(int(self.steps)):
            v.rate(int(self.steps/self.tm))
            graph1.plot(self.listat[i], self.listaO1[i])
            graph2.plot(self.listat[i], self.listaO2[i])
            graph3.plot(self.listat[i], self.listaO3[i])
            graph1a.plot(self.listat[i],self.o1[i]) # omega 1 - svjetlo plavo
            graph2a.plot(self.listat[i],self.o2[i]) # omega 2 - zeleno
            graph3a.plot(self.listat[i],self.o3[i]) # omega 3 - crveno
        while True:
            v.rate(10)

h = zvrk(1,2,1,0,3,10,15,10)
h.calc()
h.int()
h.sim()
