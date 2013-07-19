# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:44:41 2011

@author: hanliumaozhi
"""

__metaclass__ = type

from chaco.api import ArrayPlotData, Plot, Legend,HPlotContainer
from enable.component_editor import ComponentEditor
from traits.api import HasTraits, Instance, Float, Button
from traitsui.api import Group, Item, View
import numpy as np
from scipy.optimize import leastsq, fsolve
from math import log
from math import e


def f(x):
    rf = float(x[0])
    rs = float(x[1])
    return [
        (rf/(rg13+rs))*((rg13+rs+rf)/(rg2+rs+rf)*es2-es13)-2,
        (rf/(rg12+rs))*((rg12+rs+rf)/(rg2+rs+rf)*es2-es12)-1,       
        ]

def calcc(x):
    r1=r2=r3=x[0]
    t13=x[8]
    t12=x[4]
    global rg13
    global rg12
    global es12
    global es13
    global es2
    global rg2
    rg13=(r1*t13)/(r1+t13)
    rg12=(r1*t12)/(r1+t12)
    es13=(t13)/(r1+t13)
    es12=(t12)/(r1+t12)
    es2=(r3)/(r2+r3)
    rg2=(r2*r3)/(r2+r3)
    
    result=fsolve(f, [1,1])
    return result

def fun(x, p):
    a, b = p
    return a*x + b

def residuals(p, x, y):
    return fun(x, p) - y
 
    
def firstcalc(x, y):
    global xx
    global yy
    xx=[]
    yy=[] 
    for i in xrange(len(x)):
        xx.append((1.0/(x[i]+273.0)-(1.0)/298.0))
        yy.append(log(1.0*y[i]))
    x1=np.array([xx[0],xx[1],xx[2],xx[3],xx[4],xx[5],xx[6],xx[7],xx[8]],dtype=float)
    y1=np.array([yy[0],yy[1],yy[2],yy[3],yy[4],yy[5],yy[6],yy[7],yy[8]],dtype=float)      
    r=leastsq(residuals, [3000, 1], args=(x1, y1))
    f=open("1.txt","w")
    for i in xrange(9):
        strrr= str(xx[i]) + " " + str(yy[i]) + '\n'
        f.write(strrr)
    f.write(str(r[0][0]))
    f.close()
    return r

class calc(HasTraits):
    
    m1 = Float(25.0)
    t1 = Float(2.44)
    m2 = Float(30.0)
    t2 = Float(2.08)
    m3 = Float(35.0)
    t3 = Float(1.80)
    m4 = Float(40.0)
    t4 = Float(1.56)
    m5 = Float(45.0)
    t5 = Float(1.36)
    m6 = Float(50.0)
    t6 = Float(1.20)
    m7 = Float(55.0)
    t7 = Float(1.06)
    m8 = Float(60.0)
    t8 = Float(0.93)
    m9 = Float(65.0)
    t9 = Float(0.81)
    ct1 = Float
    ct2 = Float
    ct3 = Float
    ct4 = Float
    ct5 = Float
    ct6 = Float
    ct7 = Float
    ct8 = Float
    ct9 = Float
    rff= Float
    rss= Float
    drg12 = Float
    drg13 = Float
    des12 = Float
    des13 = Float
    drg2 = Float
    des2 = Float
    dbx1 = Float
    dbx2 = Float
    dbx3 = Float
    dbx4 = Float
    dbx5 = Float
    dbx6 = Float
    dbx7 = Float
    dbx8 = Float
    dbx9 = Float
    dby1 = Float
    dby2 = Float
    dby3 = Float
    dby4 = Float
    dby5 = Float
    dby6 = Float
    dby7 = Float
    dby8 = Float
    dby9 = Float
    dbn = Float
    dbk = Float
    dv1 = Float
    dv2 = Float
    dv3 = Float
    dv4 = Float
    dv5 = Float
    dv6 = Float
    dv7 = Float
    dv8 = Float
    dv9 = Float
    
    x=[]
    y=[]
    xrt=[]
    tt1 = Button()

    plot = Instance(HPlotContainer)

    traits_view = View(Group(Group(
        Group(
            Item(name='m1',label= u"温度1"),
            Item(name='t1',label= u"R1")
            ),
        Group(
            Item(name = 'm2',label= u"温度2"),
            Item(name = 't2',label= u"R2")
            ),
        Group(
            Item(name = 'm3',label= u"温度3"),
            Item(name = 't3',label= u"R3")
            ),
        Group(
            Item(name = 'm4',label= u"温度4"),
            Item(name = 't4',label= u"R4")
            ),
        Group(
            Item(name = 'm5',label= u"温度5"),
            Item(name = 't5',label= u"R5")
            ),
        Group(
            Item(name = 'm6',label= u"温度6"),
            Item(name = 't6',label= u"R6"),
            ),
        Group(
            Item(name = 'm7',label= u"温度7"),
            Item(name = 't7',label= u"R7"),
            ),
        Group(
            Item(name = 'm8',label= u"温度8"),
            Item(name = 't8',label= u"R8"),
            ),
        Group(
            Item(name = 'm9',label= u"温度9"),
            Item(name = 't9',label= u"R9"),
            ),
        Item('tt1', label=u"输入完成",show_label=False),
        orientation= 'horizontal',
        label = u'输入数据',
        show_border = True
        ),
        Group(Group(Item('plot', editor=ComponentEditor(), show_label=False),
            orientation = "vertical",            
            label = u'数据处理',
        show_border = True)),
        label=u'第一页'),
        Group(Group(
            Group(
                Item(name='m1',label= u"温度1"),
                Item(name='ct1',label= u"R1")
                ),
            Group(
                Item(name = 'm2',label= u"温度2"),
                Item(name = 'ct2',label= u"R2")
                ),
            Group(
                Item(name = 'm3',label= u"温度3"),
                Item(name = 'ct3',label= u"R3")
                ),
            Group(
                Item(name = 'm4',label= u"温度4"),
                Item(name = 'ct4',label= u"R4")
                ),
            Group(
                Item(name = 'm5',label= u"温度5"),
                Item(name = 'ct5',label= u"R5")
                ),
            Group(
                Item(name = 'm6',label= u"温度6"),
                Item(name = 'ct6',label= u"R6"),
                ),
            Group(
                Item(name = 'm7',label= u"温度7"),
                Item(name = 'ct7',label= u"R7"),
                ),
            Group(
                Item(name = 'm8',label= u"温度8"),
                Item(name = 'ct8',label= u"R8"),
                ),
            Group(
                Item(name = 'm9',label= u"温度9"),
                Item(name = 'ct9',label= u"R9"),
                ),
                orientation= 'horizontal',
                label = u'理论数据',
                show_border = True
                ),
            Group(
                Item(name='rff',label= u'Rf'),
                Item(name='rss',label= u'Rs'),
            ),
            Group(Group(
                Item(name='drg12',label= u'Rg12'),
                Item(name='drg13',label= u'Rg13'),
                ),
                Group(
                Item(name='des12',label= u'Es12'),
                Item(name='des13',label= u'Es13')
                ),
                Group(
                Item(name='drg2',label= u'Rg2'),
                Item(name='des2',label= u'Es2'),
                ),
            orientation= 'horizontal',
            ),
        Group(
            Group(
                Item(name='dbx1',label= u"X1"),
                Item(name='dby1',label= u"Y1")
                ),
            Group(
                Item(name = 'dbx2',label= u"X2"),
                Item(name = 'dby2',label= u"Y2")
                ),
            Group(
                Item(name = 'dbx3',label= u"X3"),
                Item(name = 'dby3',label= u"Y3")
                ),
            Group(
                Item(name = 'dbx4',label= u"X4"),
                Item(name = 'dby4',label= u"Y4")
                ),
            Group(
                Item(name = 'dbx5',label= u"X5"),
                Item(name = 'dby5',label= u"Y5")
                ),
            Group(
                Item(name = 'dbx6',label= u"X6"),
                Item(name = 'dby6',label= u"Y6"),
                ),
            Group(
                Item(name = 'dbx7',label= u"X7"),
                Item(name = 'dby7',label= u"Y7"),
                ),
            Group(
                Item(name = 'dbx8',label= u"X8"),
                Item(name = 'dby8',label= u"Y8"),
                ),
            Group(
                Item(name = 'dbx9',label= u"X9"),
                Item(name = 'dby9',label= u"Y9"),
                ),
                orientation= 'horizontal',
                label = u'最小二乘法求bn',
                show_border = True
                ),
            Group(
                Item(name='dbn',label= u'Bn(K)斜率'),
                Item(name='dbk',label= u'截距')
                ),
            Group(
            Group(
                Item(name='m1',label= u"温度1"),
                Item(name='dv1',label= u"V1")
                ),
            Group(
                Item(name = 'm2',label= u"温度2"),
                Item(name = 'dv2',label= u"V2")
                ),
            Group(
                Item(name = 'm3',label= u"温度3"),
                Item(name = 'dv3',label= u"V3")
                ),
            Group(
                Item(name = 'm4',label= u"温度4"),
                Item(name = 'dv4',label= u"V4")
                ),
            Group(
                Item(name = 'm5',label= u"温度5"),
                Item(name = 'dv5',label= u"V5")
                ),
            Group(
                Item(name = 'm6',label= u"温度6"),
                Item(name = 'dv6',label= u"V6"),
                ),
            Group(
                Item(name = 'm7',label= u"温度7"),
                Item(name = 'dv7',label= u"V7"),
                ),
            Group(
                Item(name = 'm8',label= u"温度8"),
                Item(name = 'dv8',label= u"V8"),
                ),
            Group(
                Item(name = 'm9',label= u"温度9"),
                Item(name = 'dv9',label= u"V9"),
                ),
                orientation= 'horizontal',
                label = u'温度传感器的电压-温度 关系',
                show_border = True
                ),
            label = u'第二页'
        ),
        width=800, height=600, resizable=True,
        title=u"物理实验 by：hanliumaozhi"
                    )

    def __init__(self):
        super(calc, self).__init__()
        self.addele()
        self.plotdata = ArrayPlotData(x = self.x, y = self.y, xrt=self.xrt, xx=xx, yy=yy,tt=self.ttt)
        plot1 =  Plot(self.plotdata)
        plot1.plot(("x", "y"),type="line",color="blue",name='1')
        plot1.plot(("x", "y"),type="scatter",color="blue", marker = 'circle', marker_size = 2,name='1')
        plot1.plot(("x", "xrt"),type="line",color="red",name='2')
        plot1.plot(("x", "xrt"),type="scatter",color="red", marker = 'circle', marker_size = 2,name='2')
        plot2 = Plot(self.plotdata)
        plot2.plot(("xx", "yy"),type="line",color="blue")
        plot2.plot(("xx","yy"),type="scatter",color="blue")
        plot3 = Plot(self.plotdata)
        plot3.plot(("x", "tt"),type="line",color="blue")
        plot3.plot(("x","tt"),type="scatter",color="blue")
        container = HPlotContainer(plot1,plot2,plot3)        
        self.plot= container
        
        legend= Legend(padding=10, align="ur")
        legend.plots = plot1.plots
        plot1.overlays.append(legend)
        

        
        
        
    def _tt1_fired(self):
        self.addele()
        self.plotdata.del_data("x")
        self.plotdata.del_data("y")
        self.plotdata.del_data("xrt")
        self.plotdata.del_data("xx")
        self.plotdata.del_data("yy")
        self.plotdata.del_data("tt")
        self.plotdata.set_data("x",self.x)
        self.plotdata.set_data("y",self.y)
        self.plotdata.set_data("xrt",self.xrt)
        self.plotdata.set_data("xx",xx)
        self.plotdata.set_data("yy",yy)
        self.plotdata.set_data("tt",self.ttt)
        plot1 =  Plot(self.plotdata)
        plot1.plot(("x", "y"),type="line",color="blue",name='1')
        plot1.plot(("x", "y"),type="scatter",color="blue", marker = 'circle', marker_size = 2,name='1')
        plot1.plot(("x", "xrt"),type="line",color="red",name='2')
        plot1.plot(("x", "xrt"),type="scatter",color="red", marker = 'circle', marker_size = 2,name='2')
        plot2 = Plot(self.plotdata)
        plot2.plot(("xx", "yy"),type="line",color="blue")
        plot2.plot(("xx","yy"),type="scatter",color="blue")
        plot3 = Plot(self.plotdata)
        plot3.plot(("x", "tt"),type="line",color="blue")
        plot3.plot(("x","tt"),type="scatter",color="blue")
        container = HPlotContainer(plot1,plot2,plot3)        
        self.plot= container
        
        legend= Legend(padding=10, align="ur")
        legend.plots = plot1.plots
        plot1.overlays.append(legend)
        

        

            
        
        

        
    def addele(self):
        self.xrt=[]
        self.x=[self.m1,self.m2,self.m3,self.m4,self.m5,self.m6,self.m7,self.m8,self.m9]
        self.y=[self.t1,self.t2,self.t3,self.t4,self.t5,self.t6,self.t7,self.t8,self.t9]
        self.r=firstcalc(self.x, self.y)
        for i in self.x:
            self.xrt.append((e**(self.r[0][1]))*(e**(self.r[0][0]*(1.0/(i+273)-(1.0)/298))))
        self.rfs=calcc(self.y)
        self.ttt=[]
        for i in self.y:
            self.ttt.append((self.rfs[0]/((self.y[0]*i)/(self.y[0]+i)+self.rfs[1]))*(((self.y[0]*i)/(self.y[0]+i)+self.rfs[0]+self.rfs[1])/(rg2+self.rfs[0]+self.rfs[1])*es2-(i)/(self.y[0]+i)))
        f=open("t.txt","w")
        for i in self.xrt:
            f.write(str(i))
            f.write("\n")
        f.write(str(self.r[0][0]))
        f.write("\n")
        f.write(str(self.r[0][1]))
        f.close()
        self.rff=self.rfs[0]
        self.rss=self.rfs[1]
        self.ct1=self.xrt[0]
        self.ct2=self.xrt[1]
        self.ct3=self.xrt[2]
        self.ct4=self.xrt[3]
        self.ct5=self.xrt[4]
        self.ct6=self.xrt[5]
        self.ct7=self.xrt[6]
        self.ct8=self.xrt[7]
        self.ct9=self.xrt[8]
        self.drg12 = rg12
        self.drg13 = rg13
        self.des12 = es12
        self.des13 = es13
        self.drg2 = rg2
        self.des2 = es2
        self.dbx1 = round(xx[0],5)
        self.dbx2 = round(xx[1],5)
        self.dbx3 = round(xx[2],5)
        self.dbx4 = round(xx[3],5)
        self.dbx5 = round(xx[4],5)
        self.dbx6 = round(xx[5],5)
        self.dbx7 = round(xx[6],5)
        self.dbx8 = round(xx[7],5)
        self.dbx9 = round(xx[8],5)
        self.dby1 = round(yy[0],5)
        self.dby2 = round(yy[1],5)
        self.dby3 = round(yy[2],5)
        self.dby4 = round(yy[3],5)
        self.dby5 = round(yy[4],5)
        self.dby6 = round(yy[5],5)
        self.dby7 = round(yy[6],5)
        self.dby8 = round(yy[7],5)
        self.dby9 = round(yy[8],5)
        self.dbn = self.r[0][0]
        self.dbk = self.r[0][1]
        self.dv1= round(self.ttt[0],2)
        self.dv2= round(self.ttt[1],2)
        self.dv3= round(self.ttt[2],2)
        self.dv4= round(self.ttt[3],2)
        self.dv5= round(self.ttt[4],2)
        self.dv6= round(self.ttt[5],2)
        self.dv7= round(self.ttt[6],2)
        self.dv8= round(self.ttt[7],2)
        self.dv9= round(self.ttt[8],2)
        
        
        
            
            

if __name__ == "__main__":
    sam=calc()    
    sam.configure_traits()
