from os import remove, times
import tkinter as tk
from typing import Sized
from matplotlib import figure
import matplotlib.pyplot as plt
from tkinter.constants import ALL, ANCHOR, BOTTOM, END, LEFT, RIDGE, RIGHT, SOLID, TOP, VERTICAL, Y
from tkinter import Canvas, Listbox, Variable, colorchooser
from tkinter import ttk
from tkinter import messagebox as msg
from tkinter import filedialog
from PIL import ImageTk, Image 
import cv2 
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy.lib.function_base import delete
import copy

fileName=[]
filename_1=[]
frames=[]
count=0


class MyVideo:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            print("Unable to open video source", video_source)
        
        self.w = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.f=1000/self.vid.get(cv2.CAP_PROP_FPS)

    def get_read(self):
        try:
            if self.vid.isOpened():
                ret, self.frame = self.vid.read()
                return (ret, cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        except:
            return False,None



def mouse_fn(event,x,y,flags,param):
    global lower,upper,result_vid,x0,y0
    if event==cv2.EVENT_LBUTTONDOWN:
        if objectcho_1.get()==2 or objectcho_1.get()==1:
                if objectcho_2.get()==1 or objectcho_1.get()==1 : 
                    x0,y0=x,y
        color = result_vid[y, x]
        one_pixel = np.uint8([[color]])
        hsv = cv2.cvtColor(one_pixel, cv2.COLOR_RGB2HSV)
        hsv = hsv[0][0]
        lower = np.array([hsv[0]-10, 90, 100]) # 빨강색 범위 
        upper = np.array([hsv[0]+15 , 255, 255]) 
        if lower[0]<0:
            lower[0]=1
    

def s_scale():  
    global roi_s
    frame_1=vid.get_read()[1]
    cv2.putText(frame_1, "Scale", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

    roi_s=cv2.selectROI('set scale',frame_1)
    cv2.destroyWindow('set scale')
    crop()
    

def crop():
    global  roi,roi_s, Constant_x,Constant_y,START, photo_1
    frame_1=vid.get_read()[1]
    cv2.putText(frame_1, "cut image", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

    roi=cv2.selectROI('crop img',frame_1)
    eVideo.config(width=roi[2], height=roi[3])
    result_vid=frame_1[roi[1]:roi[1] + roi[3],roi[0]:roi[0] + roi[2]]
    photo_1 = ImageTk.PhotoImage(image = Image.fromarray(result_vid))
    eVideo.create_image(0, 0, image = photo_1, anchor = 'nw')
    cv2.destroyWindow('crop img')

    if objectcho_1.get()==1:
        set_Origin['state']=tk.NORMAL
        set_scale['state'] = tk.DISABLED
    elif objectcho_1.get()==2:
        set_Origin1['state']=tk.NORMAL
        set_scale1['state'] = tk.DISABLED
    
    if objectcho_1.get()==1:
        sx_w=float(str(scale_entryx1.get()))
        sy_h=float(str(scale_entryy1.get()))
        Constant_x=float(sx_w/(float(roi_s[2]*roi[2]/vid.w)))
        Constant_y=float(sy_h/(float(roi_s[3]*roi[3]/vid.h)))

    elif objectcho_1.get()==2:
        sx_w=float(str(scale_entryx2.get()))
        sy_h=float(str(scale_entryy2.get()))
        Constant_x=float(sx_w/(float(roi_s[2])))
        Constant_y=float(sy_h/(float(roi_s[3])))
    print(Constant_x,Constant_y)
    START=1
    



def selectobj():
    global roi ,upper,lower,x0,y0,result_vid
    ret,frame_1=vid.get_read()
    result_vid=frame_1[roi[1]:roi[1] + roi[3],roi[0]:roi[0] + roi[2]].copy()
    cv2.namedWindow('selectobject')
    cv2.setMouseCallback('selectobject',mouse_fn)
    while 1:
        cv2.imshow('selectobject',result_vid)
        if cv2.waitKey()==32:
            cv2.destroyWindow('selectobject')
            set_Origin['state'] = tk.DISABLED
            break
    print(x0,y0)
    print(upper,lower)
    if objectcho_1.get()==1:
        set_start['state'] = tk.NORMAL
        set_Origin['state']=tk.DISABLED
    elif objectcho_1.get()==2:
        set_start1['state'] = tk.NORMAL
        set_Origin1['state']=tk.DISABLED

def openVideo1(event=None):
    global filename_1, vid 
    filename_1 = filedialog.askopenfilename(title="Select file", filetypes=(("MP4 files", "*.mp4"),
                                                                                         ("WMV files", "*.wmv"), ("AVI files", "*.avi")))
    vid=MyVideo(filename_1)
    eVideo.config(width=vid.w, height=vid.h)
    file__name1.configure(text= filename_1)
    

def play_video():
    global vid, photo_1, reseult_vid,count,lower,upper,x0,y0
    count+=1
    datax=[]
    datay=[]
    delay=80
    
    ret, frame = vid.get_read()
     # Get a frame from the video source, and go to the next frame automatically 

    if ret:
        result_vid=frame[roi[1]:roi[1] + roi[3],roi[0]:roi[0] + roi[2]].copy()
        hsv = cv2.cvtColor(result_vid, cv2.COLOR_RGB2HSV)
        mask2 = cv2.inRange(hsv, lower, upper)
        _,gray=cv2.threshold(mask2,50,255,cv2.THRESH_BINARY)   
        gray_1=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,4)
        contours ,hie=cv2.findContours(gray_1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        try:
            contour=contours[0]
            topmost = list(contour[contour[:,:,1].argmin()][0])
            for i in contours:
                M=cv2.moments(i)
                cX=int(M['m10']/ M['m00'])
                cY=int(M['m01']/ M['m00'])   
                datax.append((cX))
                datay.append((cY))

            CX=round(max(datax)-int(x0),1)
            #CY=round(max(datay)-int(y0),1)
            CY=round(int(topmost[1])-int(y0),1)
           
            if objectcho_1.get()==1:
                databox_1.insert('end',f'{round(vid.f*count,1)} {CX},{CY}')
                cv2.circle(result_vid,(max(datax),max(datay)),5,(255,0,0),5)
            elif objectcho_1.get()==2:
                if objectcho_2.get()==1:   
                    databox_2.insert('end',f'{round(vid.f*count,1)} {(CX)},{(CY)}')
                    cv2.circle(result_vid,(max(datax),max(datay)),5,(255,0,0),5)
                elif objectcho_2.get()==2:
                    databox_3.insert('end',f'{round(vid.f*count,1)} {(CX)},{(CY)}')
                    cv2.circle(result_vid,(max(datax),max(datay)),5,(255,0,0),5)
        except:
            print('no',count)
            pass
        photo_1 = ImageTk.PhotoImage(image = Image.fromarray(result_vid))
        eVideo.create_image(0, 0, image = photo_1, anchor = 'nw')
        win.after(delay, play_video)
    else:
        end()

def reset():
    global count
    print('clear')
    count=0
    databox_1.delete(0,'end')
    databox_2.delete(0,'end')
    databox_3.delete(0,'end')
    set_scale['state'] = tk.NORMAL
    set_start['state'] = tk.DISABLED
    set_Origin['state'] = tk.DISABLED
    set_scale1['state'] = tk.NORMAL
    set_start1['state'] = tk.DISABLED
    set_Origin1['state'] = tk.DISABLED

def open_button():
    set_scale['state'] = tk.NORMAL
    set_scale1['state'] = tk.NORMAL

        
def end():
    if objectcho_1.get()==1:    
        set_start['state'] = tk.DISABLED
    elif objectcho_1.get()==2:
        set_start1['state'] = tk.DISABLED
    count=0

def second():
    global vid, count
    vid=MyVideo(filename_1)
    count=0
    reset1['state']=tk.DISABLED
    set_Origin1['state']=tk.NORMAL

def __help1():
    info = '''1.영상을 불러옵니다 
    2.단일 물체와 다중 물체를 구분하여 analyze start를 누릅니다
    3.켜진 버튼을 순서대로 누릅니다
    4.드래그를 이용하여 스케일을 설정합니다
    5.불필요한 물체와 비슷한 색이 안 들어가도록 영상을 자릅니다
    6.움직임을 분석할 물체를 클릭합니다
    7.video start를 누르고 영상이 끝나면 분석탭에서 분석합니다'''
    msg.showinfo("도움말", info)


#####################여기까지 트래킹#######################

#####################마우스포인팅##########################
def analyzebtn():
    global Canvas, graphframe, aax, aay, time, laaa, START
    graphframe = tk.Frame(tab2)
    graphframe.pack()
    datalist = []
    
    if START==1:
        if objectcho_1.get()==1:
            data=databox_1
        elif objectcho_1.get()==2:
            data_2=databox_2
            data_next_2=databox_3
    else:
        pass

    if whatislove.get() == 1 :
        if objectcho.get() == 1 or objectcho_1.get()==1:
            for ii in range(data.size()):
                datalist.append(data.get(0,ii))
            origin = list(datalist[0])
            origin1 = ''.join(origin)
            t0, x0y0 = origin1.split(' ')
            x0, y0 = x0y0.split(',')
            tnstr = list(datalist.pop())
            time=[]
            aax=[]
            aay=[]
            for aa in range(len(tnstr)):
                tnp = tnstr[aa:aa+1]
                tnpstr = ''.join(tnp)
                t,xy = tnpstr.split(' ')
                x,y = xy.split(',')
                time.append((float(t)-float(t0))*(1/1000))
                aax.append((int(x)-int(x0))*Constant_x)
                aay.append((int(y0)-int(y))*Constant_y)
            while float(time[-1]) == float(0):
                time.pop()
                aax.pop()
                aay.pop()
            x1 = time
            y1 = aay
            figure = plt.figure(figsize=(10, 7), dpi=100) 
            linearmodel_1 = np.polyfit(x1,y1,2)
            linearx= np.arange(0,time[-1])
            linearfitting1 = np.poly1d(linearmodel_1)
            f1 = linearfitting1(time)
            x2 = time
            y2 = aax
            linearmodel_2 = np.polyfit(x2,y2,2)
            lineary= np.arange(0,time[-1])
            linearfitting2 = np.poly1d(linearmodel_2)
            f2 = linearfitting2(time)
            Radio_11 = Radio_1_2.get()
            x3 = aax
            y3 = aay
            linearmodel_3 = np.polyfit(x3,y3,2)
            lineary= np.arange(0,time[-1])
            linearfitting3 = np.poly1d(linearmodel_3)
            f3 = linearfitting3(aax)
            Radio_11 = Radio_1_2.get()
            laaa = tk.LabelFrame(move12)
            laaa.pack()
            labee = tk.Label(laaa, text= "물체의 상대적 좌표 (원점기준) \n time[s] x[m] y[m]")
            labee.pack()
            datalist111 = tk.Listbox(laaa)
            datalist111.pack()
            for n in range(len(time)):
                datalist111.insert(END, str(round(float(time[n]),3)) + " " + str(round(float(aax[n]),3)) + " " + str(round(float(aay[n]),3)))
            if Radio_11 == 1:
                ax1 = figure.add_subplot(1,1,1)
                ax1.plot(time,aax,'*r', label = "data")
                ax1.legend()
                tx = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n a = " + str(round(linearmodel_2[0],3)) + ",b = " + str(round(linearmodel_2[1],3)) + ",c = " + str(round(linearmodel_2[2],3)))
                ax1.set_xlabel("time [s]")
                ax1.set_ylabel("Distance between origin and click point of x [m]")
                tx.pack()
                
            elif Radio_11 == 2:
                ax2 = figure.add_subplot(1,1,1)
                ax2.plot(time,aay,'*r', label = "data")
                ax2.legend()
                ty = tk.Label(graphframe, text=" y= ax^2 + bx + c" + "\n a = " + str(round(linearmodel_1[0],3)) + ",b = " + str(round(linearmodel_1[1],3)) + ",c = " + str(round(linearmodel_1[2],3)))
                ax2.set_xlabel("time [s]")
                ax2.set_ylabel("Distance between origin and click point of y [m]")
                ty.pack()
            else:
                ax2 = figure.add_subplot(1,1,1)
                ax2.plot(aax,aay,'*r', label = "data")
                ax2.legend()
                xy = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n a = " + str(round(linearmodel_3[0],3)) + ",b = " + str(round(linearmodel_3[1],3)) + ",c = " + str(round(linearmodel_3[2],3)))
                ax2.set_xlabel("Distance between origin and click point of x [m]")
                ax2.set_ylabel("Distance between origin and click point y [m]")
                xy.pack()
            Canvas = FigureCanvasTkAgg(figure, master=graphframe)
            Canvas.draw()
            Canvas.get_tk_widget().pack()
        elif objectcho.get() == 2 or objectcho_1.get()==2:
                
                ##첫번쨰 물체
                datalist = []
                for ii in range(data_2.size()):
                    datalist.append(data_2.get(0,ii))
                origin = list(datalist[0])
                origin1 = ''.join(origin)
                t0, x0y0 = origin1.split(' ')
                x0, y0 = x0y0.split(',')
                tnstr = list(datalist.pop())
                time=[]
                aax=[]
                aay=[]
                for aa in range(len(tnstr)):
                    tnp = tnstr[aa:aa+1]
                    tnpstr = ''.join(tnp)
                    t,xy = tnpstr.split(' ')
                    x,y = xy.split(',')
                    time.append((float(t)-float(t0))*(1/1000))
                    aax.append((int(x)-int(x0))*Constant_x)
                    aay.append((int(y0)-int(y))*Constant_y)
                while float(time[-1]) == float(0):
                    time.pop()
                    aax.pop()
                    aay.pop()
                x1 = time
                y1 = aay
                linearmodel_1 = np.polyfit(x1,y1,2)
                linearx= np.arange(0,time[-1])
                linearfitting1 = np.poly1d(linearmodel_1)
                f1 = linearfitting1(time)
                x2 = time
                y2 = aax
                linearmodel_2 = np.polyfit(x2,y2,2)
                lineary= np.arange(0,time[-1])
                linearfitting2 = np.poly1d(linearmodel_2)
                f2 = linearfitting2(time)
                Radio_11 = Radio_1_2.get()
                x3 = aax
                y3 = aay
                linearmodel_3 = np.polyfit(x3,y3,2)
                lineary= np.arange(0,time[-1])
                linearfitting3 = np.poly1d(linearmodel_3)
                f3 = linearfitting3(aax)
                ##두번째물체
                datalist12345 = []
                for ii in range(data_next_2.size()):
                    datalist12345.append(data_next_2.get(0,ii))
                origin123 = list(datalist12345[0])
                origin11 = ''.join(origin123)
                t01, x0y01 = origin11.split(' ')
                x01, y01 = x0y01.split(',')
                tnstr1 = list(datalist12345.pop())
                time1=[]
                aax1=[]
                aay1=[]
                for aa in range(len(tnstr1)):
                    tnp1 = tnstr1[aa:aa+1]
                    tnpstr1 = ''.join(tnp1)
                    t,xy = tnpstr1.split(' ')
                    x,y = xy.split(',')
                    time1.append((float(t)-float(t0))*(1/1000))
                    aax1.append((int(x)-int(x0))*Constant_x)
                    aay1.append((int(y0)-int(y))*Constant_y)
                while float(time1[-1]) == float(0):
                    time1.pop()
                    aax1.pop()
                    aay1.pop()
                ##피팅
                x11 = time1
                y11 = aay1
                linearmodel_11 = np.polyfit(x11,y11,2)
                linearx= np.arange(0,time1[-1])
                linearfitting11 = np.poly1d(linearmodel_11)
                f11 = linearfitting11(time1)
                x21 = time1
                y21 = aax1
                linearmodel_21 = np.polyfit(x21,y21,2)
                lineary11= np.arange(0,time1[-1])
                linearfitting21 = np.poly1d(linearmodel_21)
                f21 = linearfitting21(time1)
                Radio_111 = Radio_1_2.get()
                x31= aax1
                y31 = aay1
                linearmodel_31 = np.polyfit(x31,y31,2)
                lineary11= np.arange(0,time1[-1])
                linearfitting31 = np.poly1d(linearmodel_31)
                f31 = linearfitting31(aax1)
                Radio_11 = Radio_1_2.get()
                ##리스트박스제작
                laaa = tk.LabelFrame(move12)
                laaa.pack()
                labee = tk.Label(laaa, text= "물체의 상대적 좌표 (원점기준) \n time[s] x[m] y[m]")
                labee.pack()
                datalist111 = tk.Listbox(laaa)
                datalist111.pack(side='left')
                datalist1111 = tk.Listbox(laaa)
                datalist1111.pack(side='left')
                for n in range(len(time)):
                    datalist111.insert(END, str(round(float(time[n]),3)) + " " + str(round(float(aax[n]),3)) + " " + str(round(float(aay[n]),3)))
                for n in range(len(time)):
                    datalist1111.insert(END, str(round(float(time1[n]),3)) + " " + str(round(float(aax1[n]),3)) + " " + str(round(float(aay1[n]),3)))
                ##플랏
                figure = plt.figure(figsize=(10, 7), dpi=100) 
                if Radio_11 == 1:
                    ax1 = figure.add_subplot(2,1,1)
                    ax11 = figure.add_subplot(2,1,2)
                    ax1.plot(time,aax,'*r', label = "data1")
                    ax11.plot(time1,aax1, '*g', label = "data2")
                    ax1.legend()
                    ax11.legend()
                    tx = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n 첫번째 물체 : a = " + str(round(linearmodel_2[0],3)) + ",b = " + str(round(linearmodel_2[1],3)) + ",c = " + str(round(linearmodel_2[2],3)) + "\n 두번째 물체 : a = " + str(round(linearmodel_21[0],3)) + ",b = " + str(round(linearmodel_21[1],3)) + ",c = " + str(round(linearmodel_21[2],3)))
                    ax1.set_xlabel("time [s]")
                    ax1.set_ylabel("Distance between origin and click point of x [m]")
                    ax11.set_xlabel("time [s]")
                    ax11.set_ylabel("Distance between origin and click point of x [m]")
                    tx.pack()
                elif Radio_11 == 2:
                    ax2 = figure.add_subplot(2,1,1)
                    ax22= figure.add_subplot(2,1,2)
                    ax2.plot(time,aay,'*r', label = "data1")
                    ax22.plot(time1,aay1,'*g', label = "data2")
                    ax2.legend()
                    ax22.legend()
                    ty = tk.Label(graphframe, text=" y= ax^2 + bx + c" + "\n 첫번째 물체 : a = " + str(round(linearmodel_1[0],3)) + ",b = " + str(round(linearmodel_1[1],3)) + ",c = " + str(round(linearmodel_1[2],3))+ "\n 두번째 물체 : a = " + str(round(linearmodel_11[0],3)) + ",b = " + str(round(linearmodel_11[1],3)) + ",c = " + str(round(linearmodel_11[2],3)))
                    ax2.set_xlabel("time [s]")
                    ax2.set_ylabel("Distance between origin and click point of y [m]")
                    ax22.set_xlabel("time [s]")
                    ax22.set_ylabel("Distance between origin and click point of y [m]")
                    ty.pack()
                else:
                    ax2 = figure.add_subplot(2,1,1)
                    ax22 = figure.add_subplot(2,1,2)
                    ax2.plot(aax,aay,'*r', label = "data_2")
                    ax22.plot(aax1,aay1,'*g', label = "data2")
                    ax2.legend()
                    ax22.legend()
                    xy = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n 첫번째 물체 : a = " + str(round(linearmodel_3[0],3)) + ",b = " + str(round(linearmodel_3[1],3)) + ",c = " + str(round(linearmodel_3[2],3)) + "\n 두번째 물체 : a = " + str(round(linearmodel_31[0],3)) + ",b = " + str(round(linearmodel_31[1],3)) + ",c = " + str(round(linearmodel_31[2],3)))
                    ax2.set_xlabel("Distance between origin and click point of x [m]")
                    ax2.set_ylabel("Distance between origin and click point y [m]")
                    ax22.set_xlabel("Distance between origin and click point of x [m]")
                    ax22.set_ylabel("Distance between origin and click point y [m]")
                    xy.pack()
                Canvas = FigureCanvasTkAgg(figure, master=graphframe)
                Canvas.draw()
                Canvas.get_tk_widget().pack()
    if whatislove.get() == 2:
        if objectcho.get() == 1 or objectcho_1.get()==1:
            
            for ii in range(data.size()):
                datalist.append(data.get(0,ii))
            origin = list(datalist[0])
            origin1 = ''.join(origin)
            t0, x0y0 = origin1.split(' ')
            x0, y0 = x0y0.split(',')
            tnstr = list(datalist.pop())
            time=[]
            aax=[]
            aay=[]
            for aa in range(len(tnstr)):
                tnp = tnstr[aa:aa+1]
                tnpstr = ''.join(tnp)
                t,xy = tnpstr.split(' ')
                x,y = xy.split(',')
                time.append((float(t)-float(t0))*(1/1000))
                aax.append((int(x)-int(x0))*Constant_x)
                aay.append((int(y0)-int(y))*Constant_y)
            while float(time[-1]) == float(0):
                time.pop()
                aax.pop()
                aay.pop()
            x1 = time
            y1 = aay
            figure = plt.figure(figsize=(10, 7), dpi=100) 
            linearmodel_1 = np.polyfit(x1,y1,2)
            linearx= np.arange(0,time[-1])
            linearfitting1 = np.poly1d(linearmodel_1)
            f1 = linearfitting1(time)
            x2 = time
            y2 = aax
            linearmodel_2 = np.polyfit(x2,y2,2)
            lineary= np.arange(0,time[-1])
            linearfitting2 = np.poly1d(linearmodel_2)
            f2 = linearfitting2(time)
            Radio_11 = Radio_1_2.get()
            x3 = aax
            y3 = aay
            linearmodel_3 = np.polyfit(x3,y3,2)
            lineary= np.arange(0,time[-1])
            linearfitting3 = np.poly1d(linearmodel_3)
            f3 = linearfitting3(aax)
            Radio_11 = Radio_1_2.get()
            laaa = tk.LabelFrame(move12)
            laaa.pack()
            labee = tk.Label(laaa, text= "물체의 상대적 좌표 (원점기준) \n time[s] x[m] y[m]")
            labee.pack()
            datalist111 = tk.Listbox(laaa)
            datalist111.pack()
            for n in range(len(time)):
                datalist111.insert(END, str(round(float(time[n]),3)) + " " + str(round(float(aax[n]),3)) + " " + str(round(float(aay[n]),3)))
            if Radio_11 == 1:
                ax1 = figure.add_subplot(1,1,1)
                ax1.plot(time,f2, 'r-', label = "fitting")
                ax1.legend()
                tx = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n a = " + str(round(linearmodel_2[0],3)) + ",b = " + str(round(linearmodel_2[1],3)) + ",c = " + str(round(linearmodel_2[2],3)))
                ax1.set_xlabel("time [s]")
                ax1.set_ylabel("Distance between origin and click point of x [m]")
                tx.pack()
                
            elif Radio_11 == 2:
                ax2 = figure.add_subplot(1,1,1)
                ax2.plot(time,f1, 'r-', label = "fitting")
                ax2.legend()
                ty = tk.Label(graphframe, text=" y= ax^2 + bx + c" + "\n a = " + str(round(linearmodel_1[0],3)) + ",b = " + str(round(linearmodel_1[1],3)) + ",c = " + str(round(linearmodel_1[2],3)))
                ax2.set_xlabel("time [s]")
                ax2.set_ylabel("Distance between origin and click point of y [m]")
                ty.pack()
            else:
                ax2 = figure.add_subplot(1,1,1)
                ax2.plot(aax,f3, 'r-', label = "fitting")
                ax2.legend()
                xy = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n a = " + str(round(linearmodel_3[0],3)) + ",b = " + str(round(linearmodel_3[1],3)) + ",c = " + str(round(linearmodel_3[2],3)))
                ax2.set_xlabel("Distance between origin and click point of x [m]")
                ax2.set_ylabel("Distance between origin and click point y [m]")
                xy.pack()
            Canvas = FigureCanvasTkAgg(figure, master=graphframe)
            Canvas.draw()
            Canvas.get_tk_widget().pack()
        elif objectcho.get() == 2 or objectcho_1.get()==2:
                ##첫번쨰 물체
                datalist = []
                for ii in range(data_2.size()):
                    datalist.append(data_2.get(0,ii))
                origin = list(datalist[0])
                origin1 = ''.join(origin)
                t0, x0y0 = origin1.split(' ')
                x0, y0 = x0y0.split(',')
                tnstr = list(datalist.pop())
                time=[]
                aax=[]
                aay=[]
                for aa in range(len(tnstr)):
                    tnp = tnstr[aa:aa+1]
                    tnpstr = ''.join(tnp)
                    t,xy = tnpstr.split(' ')
                    x,y = xy.split(',')
                    time.append((float(t)-float(t0))*(1/1000))
                    aax.append((int(x)-int(x0))*Constant_x)
                    aay.append((int(y0)-int(y))*Constant_y)
                while float(time[-1]) == float(0):
                    time.pop()
                    aax.pop()
                    aay.pop()
                x1 = time
                y1 = aay
                linearmodel_1 = np.polyfit(x1,y1,2)
                linearx= np.arange(0,time[-1])
                linearfitting1 = np.poly1d(linearmodel_1)
                f1 = linearfitting1(time)
                x2 = time
                y2 = aax
                linearmodel_2 = np.polyfit(x2,y2,2)
                lineary= np.arange(0,time[-1])
                linearfitting2 = np.poly1d(linearmodel_2)
                f2 = linearfitting2(time)
                Radio_11 = Radio_1_2.get()
                x3 = aax
                y3 = aay
                linearmodel_3 = np.polyfit(x3,y3,2)
                lineary= np.arange(0,time[-1])
                linearfitting3 = np.poly1d(linearmodel_3)
                f3 = linearfitting3(aax)
                ##두번째물체
                datalist12345 = []
                for ii in range(data_next_2.size()):
                    datalist12345.append(data_next_2.get(0,ii))
                origin123 = list(datalist12345[0])
                origin11 = ''.join(origin123)
                t01, x0y01 = origin11.split(' ')
                x01, y01 = x0y01.split(',')
                tnstr1 = list(datalist12345.pop())
                time1=[]
                aax1=[]
                aay1=[]
                for aa in range(len(tnstr1)):
                    tnp1 = tnstr1[aa:aa+1]
                    tnpstr1 = ''.join(tnp1)
                    t,xy = tnpstr1.split(' ')
                    x,y = xy.split(',')
                    time1.append((float(t)-float(t0))*(1/1000))
                    aax1.append((int(x)-int(x0))*Constant_x)
                    aay1.append((int(y0)-int(y))*Constant_y)
                while float(time1[-1]) == float(0):
                    time1.pop()
                    aax1.pop()
                    aay1.pop()
                ##피팅
                x11 = time1
                y11 = aay1
                linearmodel_11 = np.polyfit(x11,y11,2)
                linearx= np.arange(0,time1[-1])
                linearfitting11 = np.poly1d(linearmodel_11)
                f11 = linearfitting11(time1)
                x21 = time1
                y21 = aax1
                linearmodel_21 = np.polyfit(x21,y21,2)
                lineary11= np.arange(0,time1[-1])
                linearfitting21 = np.poly1d(linearmodel_21)
                f21 = linearfitting21(time1)
                Radio_111 = Radio_1_2.get()
                x31= aax1
                y31 = aay1
                linearmodel_31 = np.polyfit(x31,y31,2)
                lineary11= np.arange(0,time1[-1])
                linearfitting31 = np.poly1d(linearmodel_31)
                f31 = linearfitting31(aax1)
                Radio_11 = Radio_1_2.get()
                ##리스트박스제작
                laaa = tk.LabelFrame(move12)
                laaa.pack()
                labee = tk.Label(laaa, text= "물체의 상대적 좌표 (원점기준) \n time[s] x[m] y[m]")
                labee.pack()
                datalist111 = tk.Listbox(laaa)
                datalist111.pack(side='left')
                datalist1111 = tk.Listbox(laaa)
                datalist1111.pack(side='left')
                for n in range(len(time)):
                    datalist111.insert(END, str(round(float(time[n]),3)) + " " + str(round(float(aax[n]),3)) + " " + str(round(float(aay[n]),3)))
                for n in range(len(time)):
                    datalist1111.insert(END, str(round(float(time1[n]),3)) + " " + str(round(float(aax1[n]),3)) + " " + str(round(float(aay1[n]),3)))
                ##플랏
                figure = plt.figure(figsize=(10, 7), dpi=100) 
                if Radio_11 == 1:
                    ax1 = figure.add_subplot(2,1,1)
                    ax11 = figure.add_subplot(2,1,2)
                    ax1.plot(time,f2, 'r-', label = "fitting1")
                    ax11.plot(time1,f21, 'g-', label = "fitting2")
                    ax1.legend()
                    ax11.legend()
                    tx = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n 첫번째 물체 : a = " + str(round(linearmodel_2[0],3)) + ",b = " + str(round(linearmodel_2[1],3)) + ",c = " + str(round(linearmodel_2[2],3)) + "\n 두번째 물체 : a = " + str(round(linearmodel_21[0],3)) + ",b = " + str(round(linearmodel_21[1],3)) + ",c = " + str(round(linearmodel_21[2],3)))
                    ax1.set_xlabel("time [s]")
                    ax1.set_ylabel("Distance between origin and click point of x [m]")
                    ax11.set_xlabel("time [s]")
                    ax11.set_ylabel("Distance between origin and click point of x [m]")
                    tx.pack()
                elif Radio_11 == 2:
                    ax2 = figure.add_subplot(2,1,1)
                    ax22= figure.add_subplot(2,1,2)
                    ax2.plot(time,f1, 'r-', label = "fitting1")
                    ax22.plot(time1,f11, 'g-', label = "fitting2")
                    ax2.legend()
                    ax22.legend()
                    ty = tk.Label(graphframe, text=" y= ax^2 + bx + c" + "\n 첫번째 물체 : a = " + str(round(linearmodel_1[0],3)) + ",b = " + str(round(linearmodel_1[1],3)) + ",c = " + str(round(linearmodel_1[2],3))+ "\n 두번째 물체 : a = " + str(round(linearmodel_11[0],3)) + ",b = " + str(round(linearmodel_11[1],3)) + ",c = " + str(round(linearmodel_11[2],3)))
                    ax2.set_xlabel("time [s]")
                    ax2.set_ylabel("Distance between origin and click point of y [m]")
                    ax22.set_xlabel("time [s]")
                    ax22.set_ylabel("Distance between origin and click point of y [m]")
                    ty.pack()
                else:
                    ax2 = figure.add_subplot(2,1,1)
                    ax22 = figure.add_subplot(2,1,2)
                    ax2.plot(aax,f3, 'r-', label = "fitting")
                    ax22.plot(aax1,f31, 'g-', label = "fitting2")
                    ax2.legend()
                    ax22.legend()
                    xy = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n 첫번째 물체 : a = " + str(round(linearmodel_3[0],3)) + ",b = " + str(round(linearmodel_3[1],3)) + ",c = " + str(round(linearmodel_3[2],3)) + "\n 두번째 물체 : a = " + str(round(linearmodel_31[0],3)) + ",b = " + str(round(linearmodel_31[1],3)) + ",c = " + str(round(linearmodel_31[2],3)))
                    ax2.set_xlabel("Distance between origin and click point of x [m]")
                    ax2.set_ylabel("Distance between origin and click point y [m]")
                    ax22.set_xlabel("Distance between origin and click point of x [m]")
                    ax22.set_ylabel("Distance between origin and click point y [m]")
                    xy.pack()
                Canvas = FigureCanvasTkAgg(figure, master=graphframe)
                Canvas.draw()
                Canvas.get_tk_widget().pack()
    if whatislove.get() == 3:
        if objectcho.get() == 1 or objectcho_1.get()==1:
            for ii in range(data.size()):
                datalist.append(data.get(0,ii))
            origin = list(datalist[0])
            origin1 = ''.join(origin)
            t0, x0y0 = origin1.split(' ')
            x0, y0 = x0y0.split(',')
            tnstr = list(datalist.pop())
            time=[]
            aax=[]
            aay=[]
            for aa in range(len(tnstr)):
                tnp = tnstr[aa:aa+1]
                tnpstr = ''.join(tnp)
                t,xy = tnpstr.split(' ')
                x,y = xy.split(',')
                time.append((float(t)-float(t0))*(1/1000))
                aax.append((int(x)-int(x0))*Constant_x)
                aay.append((int(y0)-int(y))*Constant_y)
            while float(time[-1]) == float(0):
                time.pop()
                aax.pop()
                aay.pop()
            x1 = time
            y1 = aay
            figure = plt.figure(figsize=(10, 7), dpi=100) 
            linearmodel_1 = np.polyfit(x1,y1,2)
            linearx= np.arange(0,time[-1])
            linearfitting1 = np.poly1d(linearmodel_1)
            f1 = linearfitting1(time)
            x2 = time
            y2 = aax
            linearmodel_2 = np.polyfit(x2,y2,2)
            lineary= np.arange(0,time[-1])
            linearfitting2 = np.poly1d(linearmodel_2)
            f2 = linearfitting2(time)
            Radio_11 = Radio_1_2.get()
            x3 = aax
            y3 = aay
            linearmodel_3 = np.polyfit(x3,y3,2)
            lineary= np.arange(0,time[-1])
            linearfitting3 = np.poly1d(linearmodel_3)
            f3 = linearfitting3(aax)
            Radio_11 = Radio_1_2.get()
            laaa = tk.LabelFrame(move12)
            laaa.pack()
            labee = tk.Label(laaa, text= "물체의 상대적 좌표 (원점기준) \n time[s] x[m] y[m]")
            labee.pack()
            datalist111 = tk.Listbox(laaa)
            datalist111.pack()
            for n in range(len(time)):
                datalist111.insert(END, str(round(float(time[n]),3)) + " " + str(round(float(aax[n]),3)) + " " + str(round(float(aay[n]),3)))
            if Radio_11 == 1:
                ax1 = figure.add_subplot(1,1,1)
                ax1.plot(time,aax,'*r', label = "data")
                ax1.plot(time,f2, 'r-', label = "fitting")
                ax1.legend()
                tx = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n a = " + str(round(linearmodel_2[0],3)) + ",b = " + str(round(linearmodel_2[1],3)) + ",c = " + str(round(linearmodel_2[2],3)))
                ax1.set_xlabel("time [s]")
                ax1.set_ylabel("Distance between origin and click point of x [m]")
                tx.pack()
                
            elif Radio_11 == 2:
                ax2 = figure.add_subplot(1,1,1)
                ax2.plot(time,aay,'*r', label = "data")
                ax2.plot(time,f1, 'r-', label = "fitting")
                ax2.legend()
                ty = tk.Label(graphframe, text=" y= ax^2 + bx + c" + "\n a = " + str(round(linearmodel_1[0],3)) + ",b = " + str(round(linearmodel_1[1],3)) + ",c = " + str(round(linearmodel_1[2],3)))
                ax2.set_xlabel("time [s]")
                ax2.set_ylabel("Distance between origin and click point of y [m]")
                ty.pack()
            else:
                ax2 = figure.add_subplot(1,1,1)
                ax2.plot(aax,aay,'*r', label = "data")
                ax2.plot(aax,f3, 'r-', label = "fitting")
                ax2.legend()
                xy = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n a = " + str(round(linearmodel_3[0],3)) + ",b = " + str(round(linearmodel_3[1],3)) + ",c = " + str(round(linearmodel_3[2],3)))
                ax2.set_xlabel("Distance between origin and click point of x [m]")
                ax2.set_ylabel("Distance between origin and click point y [m]")
                xy.pack()
            Canvas = FigureCanvasTkAgg(figure, master=graphframe)
            Canvas.draw()
            Canvas.get_tk_widget().pack()
        elif objectcho.get() == 2 or objectcho_1.get()==2:
                ##첫번쨰 물체
                datalist = []
                for ii in range(data_2.size()):
                    datalist.append(data_2.get(0,ii))
                origin = list(datalist[0])
                origin1 = ''.join(origin)
                t0, x0y0 = origin1.split(' ')
                x0, y0 = x0y0.split(',')
                tnstr = list(datalist.pop())
                time=[]
                aax=[]
                aay=[]
                for aa in range(len(tnstr)):
                    tnp = tnstr[aa:aa+1]
                    tnpstr = ''.join(tnp)
                    t,xy = tnpstr.split(' ')
                    x,y = xy.split(',')
                    time.append((float(t)-float(t0))*(1/1000))
                    aax.append((int(x)-int(x0))*Constant_x)
                    aay.append((int(y0)-int(y))*Constant_y)
                while float(time[-1]) == float(0):
                    time.pop()
                    aax.pop()
                    aay.pop()
                x1 = time
                y1 = aay
                linearmodel_1 = np.polyfit(x1,y1,2)
                linearx= np.arange(0,time[-1])
                linearfitting1 = np.poly1d(linearmodel_1)
                f1 = linearfitting1(time)
                x2 = time
                y2 = aax
                linearmodel_2 = np.polyfit(x2,y2,2)
                lineary= np.arange(0,time[-1])
                linearfitting2 = np.poly1d(linearmodel_2)
                f2 = linearfitting2(time)
                Radio_11 = Radio_1_2.get()
                x3 = aax
                y3 = aay
                linearmodel_3 = np.polyfit(x3,y3,2)
                lineary= np.arange(0,time[-1])
                linearfitting3 = np.poly1d(linearmodel_3)
                f3 = linearfitting3(aax)
                ##두번째물체
                datalist12345 = []
                for ii in range(data_next_2.size()):
                    datalist12345.append(data_next_2.get(0,ii))
                origin123 = list(datalist12345[0])
                origin11 = ''.join(origin123)
                t01, x0y01 = origin11.split(' ')
                x01, y01 = x0y01.split(',')
                tnstr1 = list(datalist12345.pop())
                time1=[]
                aax1=[]
                aay1=[]
                for aa in range(len(tnstr1)):
                    tnp1 = tnstr1[aa:aa+1]
                    tnpstr1 = ''.join(tnp1)
                    t,xy = tnpstr1.split(' ')
                    x,y = xy.split(',')
                    time1.append((float(t)-float(t0))*(1/1000))
                    aax1.append((int(x)-int(x0))*Constant_x)
                    aay1.append((int(y0)-int(y))*Constant_y)
                while float(time1[-1]) == float(0):
                    time1.pop()
                    aax1.pop()
                    aay1.pop()
                ##피팅
                x11 = time1
                y11 = aay1
                linearmodel_11 = np.polyfit(x11,y11,2)
                linearx= np.arange(0,time1[-1])
                linearfitting11 = np.poly1d(linearmodel_11)
                f11 = linearfitting11(time1)
                x21 = time1
                y21 = aax1
                linearmodel_21 = np.polyfit(x21,y21,2)
                lineary11= np.arange(0,time1[-1])
                linearfitting21 = np.poly1d(linearmodel_21)
                f21 = linearfitting21(time1)
                Radio_111 = Radio_1_2.get()
                x31= aax1
                y31 = aay1
                linearmodel_31 = np.polyfit(x31,y31,2)
                lineary11= np.arange(0,time1[-1])
                linearfitting31 = np.poly1d(linearmodel_31)
                f31 = linearfitting31(aax1)
                Radio_11 = Radio_1_2.get()
                ##리스트박스제작
                laaa = tk.LabelFrame(move12)
                laaa.pack()
                labee = tk.Label(laaa, text= "물체의 상대적 좌표 (원점기준) \n time[s] x[m] y[m]")
                labee.pack()
                datalist111 = tk.Listbox(laaa)
                datalist111.pack(side='left')
                datalist1111 = tk.Listbox(laaa)
                datalist1111.pack(side='left')
                for n in range(len(time)):
                    datalist111.insert(END, str(round(float(time[n]),3)) + " " + str(round(float(aax[n]),3)) + " " + str(round(float(aay[n]),3)))
                for n in range(len(time)):
                    datalist1111.insert(END, str(round(float(time1[n]),3)) + " " + str(round(float(aax1[n]),3)) + " " + str(round(float(aay1[n]),3)))
                ##플랏
                figure = plt.figure(figsize=(10, 7), dpi=100) 
                if Radio_11 == 1:
                    ax1 = figure.add_subplot(2,1,1)
                    ax11 = figure.add_subplot(2,1,2)
                    ax1.plot(time,aax,'*r', label = "data1")
                    ax1.plot(time,f2, 'r-', label = "fitting1")
                    ax11.plot(time1,aax1, '*g', label = "data2")
                    ax11.plot(time1,f21, 'g-', label = "fitting2")
                    ax1.legend()
                    ax11.legend()
                    tx = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n 첫번째 물체 : a = " + str(round(linearmodel_2[0],3)) + ",b = " + str(round(linearmodel_2[1],3)) + ",c = " + str(round(linearmodel_2[2],3)) + "\n 두번째 물체 : a = " + str(round(linearmodel_21[0],3)) + ",b = " + str(round(linearmodel_21[1],3)) + ",c = " + str(round(linearmodel_21[2],3)))
                    ax1.set_xlabel("time [s]")
                    ax1.set_ylabel("Distance between origin and click point of x [m]")
                    ax11.set_xlabel("time [s]")
                    ax11.set_ylabel("Distance between origin and click point of x [m]")
                    tx.pack()
                elif Radio_11 == 2:
                    ax2 = figure.add_subplot(2,1,1)
                    ax22= figure.add_subplot(2,1,2)
                    ax2.plot(time,aay,'*r', label = "dat1")
                    ax2.plot(time,f1, 'r-', label = "fitting1")
                    ax22.plot(time1,aay1,'*g', label = "data2")
                    ax22.plot(time1,f11, 'g-', label = "fitting2")
                    ax2.legend()
                    ax22.legend()
                    ty = tk.Label(graphframe, text=" y= ax^2 + bx + c" + "\n 첫번째 물체 : a = " + str(round(linearmodel_1[0],3)) + ",b = " + str(round(linearmodel_1[1],3)) + ",c = " + str(round(linearmodel_1[2],3))+ "\n 두번째 물체 : a = " + str(round(linearmodel_11[0],3)) + ",b = " + str(round(linearmodel_11[1],3)) + ",c = " + str(round(linearmodel_11[2],3)))
                    ax2.set_xlabel("time [s]")
                    ax2.set_ylabel("Distance between origin and click point of y [m]")
                    ax22.set_xlabel("time [s]")
                    ax22.set_ylabel("Distance between origin and click point of y [m]")
                    ty.pack()
                else:
                    ax2 = figure.add_subplot(2,1,1)
                    ax22 = figure.add_subplot(2,1,2)
                    ax2.plot(aax,aay,'*r', label = "data_2")
                    ax2.plot(aax,f3, 'r-', label = "fitting")
                    ax22.plot(aax1,aay1,'*g', label = "data2")
                    ax22.plot(aax1,f31, 'g-', label = "fitting2")
                    ax2.legend()
                    ax22.legend()
                    xy = tk.Label(graphframe, text= " y= ax^2 + bx + c" + "\n 첫번째 물체 : a = " + str(round(linearmodel_3[0],3)) + ",b = " + str(round(linearmodel_3[1],3)) + ",c = " + str(round(linearmodel_3[2],3)) + "\n 두번째 물체 : a = " + str(round(linearmodel_31[0],3)) + ",b = " + str(round(linearmodel_31[1],3)) + ",c = " + str(round(linearmodel_31[2],3)))
                    ax2.set_xlabel("Distance between origin and click point of x [m]")
                    ax2.set_ylabel("Distance between origin and click point y [m]")
                    ax22.set_xlabel("Distance between origin and click point of x [m]")
                    ax22.set_ylabel("Distance between origin and click point y [m]")
                    xy.pack()
                Canvas = FigureCanvasTkAgg(figure, master=graphframe)
                Canvas.draw()
                Canvas.get_tk_widget().pack()

class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            print("Unable to open video source", video_source)

        self.w = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)


    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)


def openVideo2(event=None):
    global frames, frameI , fileName
    fileName = filedialog.askopenfilename(filetypes=(("Video files", "*.mp4;*.flv;*.avi;*.mkv"),
                                                     ("All files", "*.*") ))
    if not fileName: return
    
    vid=MyVideoCapture(fileName)
    frames=[]
    while True:
        ret, frame = vid.get_frame()
        if not ret: break
        frames.append((ImageTk.PhotoImage(image = Image.fromarray(frame)),
                       round(vid.vid.get(cv2.CAP_PROP_POS_MSEC),1)))

    frameI=-1

    hSize.config(text=f'{int(vid.w)}x{int(vid.h)}')
    if Video.winfo_height() < vid.h:
        Video.config(width=vid.w, height=vid.h)
    nextFrame()
    change_label_2()

def change_label_2():
    file__name2.configure(text= fileName)

def nextFrame(event=None):
    global photo, frameI
    if frames:
        frameI += 1
        if frameI >= len(frames): frameI = len(frames)-1
        photo=frames[frameI][0]
        Video.itemconfig(himg, image=photo)
        hT.config(text=f'{frames[frameI][1]} ms')

def prevFrame(event=None):
    global photo, frameI
    if frames:
        frameI -= 1
        if frameI < 0: frameI=0
        photo=frames[frameI][0]
        Video.itemconfig(himg, image=photo)
        hT.config(text=f'{frames[frameI][1]} ms')

def click(event=None):
    if objectcho.get() == 1:
        if not frames: return
        data.insert('end', hT['text'].split('ms')[0]+f'{event.x},{event.y}')
        if check_auto.get(): nextFrame()
    elif objectcho.get() ==2:
        if nextobject_2.get() == 1:
                if not frames: return
                data_2.insert('end', hT['text'].split('ms')[0]+f'{event.x},{event.y}')
                if Check_auto_2.get(): nextFrame()
        elif nextobject_2.get() == 2:
                if not frames: return
                data_next_2.insert('end', hT['text'].split('ms')[0]+f'{event.x},{event.y}')
                if Check_auto_2.get(): nextFrame()

def __help2():
    info = '''1. open Video: 녹화한 파일 열기
    오류가 없는 경우 녹화한 파일의 맨처음 프레임을 보여준다.
    녹화한 프레임의 크기(폭x높이)를 보여준다.
2. next Frame: 다음 프레임으로 이동한다.
3. prev Frame: 전 프레임으로 이동한다.
4. autoNext: 체크하면 데이터를 생성하고 다음 프레임으로 이동한다.
5. 스케일설정 : 마우스 우클릭으로 스케일 박스의 좌측상단, 우측상단, 우측하단을 반드시 순서대로 클릭하여 스케일을 설정한다.
6. 분석시작 : 클릭하여 분석을 시작한다.
7. 좌표평면설정 : t-x, t-y, x-y 중 원하는 평면을 설정하여 분석을 시작할 수 있다.
8. 분석결과초기화 : 클릭하여 기존의 분석결과를 초기화하고 새로 분석을 시작할 수있다.'''
    msg.showinfo("도움말", info)


def removegraph():
    graphframe.destroy()
    laaa.destroy()

def removedata_1():
    global photo, frameI
    if frames:
        frameI = 0
        if frameI < 0: frameI=0
        photo=frames[frameI][0]
        Video.itemconfig(himg, image=photo)
        hT.config(text=f'{frames[frameI][1]} ms')
    data.delete(0,'end')
    scaledata.delete(0,'end')
    scaledata_show.delete(0,'end')
    Constant_x == float(0)
    Constant_y == float(0)

def removedata_3():
    global photo, frameI
    if frames:
        frameI = 0
        if frameI < 0: frameI=0
        photo=frames[frameI][0]
        Video.itemconfig(himg, image=photo)
        hT.config(text=f'{frames[frameI][1]} ms')
    data_2.delete(0,'end')
    scaledata_2.delete(0,'end')
    scaledata_show_2.delete(0,'end')
    data_next_2.delete(0,'end')
    Constant_x == float(0)
    Constant_y == float(0)

def removedata_2():
    global photo, frameI
    if frames:
        frameI = 0
        if frameI < 0: frameI=0
        photo=frames[frameI][0]
        Video.itemconfig(himg, image=photo)
        hT.config(text=f'{frames[frameI][1]} ms')
    data_2.delete(0,'end')
    data_next_2.delete(0,'end')
    scaledata_2.delete(0,'end')
    scaledata_show_2.delete(0,'end')
    Constant_x == float(0)
    Constant_y == float(0)

def scaleclick(event=None):
    if objectcho.get() == 1:
        if not frames: return
        scaledata.insert('end', hT['text'].split('ms')[0]+f'{event.x},{event.y}')
        if not frames: return
        scaledata_show.insert('end',f'{event.x},{event.y}')
    else:
        if not frames: return
        scaledata_2.insert('end', hT['text'].split('ms')[0]+f'{event.x},{event.y}')
        if not frames: return
        scaledata_show_2.insert('end',f'{event.x},{event.y}')

def setscalebtn():
    global Constant_x, Constant_y
    if objectcho.get() == 1:
        scaledatalist = []
        for bb in range(scaledata.size()):
            scaledatalist.append(scaledata.get(0,bb))
            print(scaledatalist)
        scale123 = list(scaledatalist.pop())
        scaleaax = []
        scaleaay = []
        for cc in range(len(scale123)):
            scale1234 = scale123[cc:cc+1]
            scale12345 = ''.join(scale1234)
            t, xy = scale12345.split(' ')
            x, y = xy.split(',')
            scaleaax.append(int(x))
            scaleaay.append(int(y))
        aaxC = float(scaleaax[1])-float(scaleaax[0])
        aayC = float(scaleaay[2])-float(scaleaay[1])
        aaxCC = float(str(scale_entryx_2.get()))
        aayCC = float(str(scale_entryy_2.get()))
        Constant_x = aaxCC/aaxC
        Constant_y = aayCC/aayC
        msg.showinfo("확인하세요", "스케일이 설정되었습니다.")
    elif objectcho.get() ==2 :
            scaledatalist = []
            for bb in range(scaledata_2.size()):
                scaledatalist.append(scaledata_2.get(0,bb))
            scale123 = list(scaledatalist.pop())
            scaleaax = []
            scaleaay = []
            for cc in range(len(scale123)):
                scale1234 = scale123[cc:cc+1]
                scale12345 = ''.join(scale1234)
                t, xy = scale12345.split(' ')
                x, y = xy.split(',')
                scaleaax.append(int(x))
                scaleaay.append(int(y))
            aaxC = float(scaleaax[1])-float(scaleaax[0])
            aayC = float(scaleaay[2])-float(scaleaay[1])
            aaxCC = float(str(scale_entryx_2.get()))
            aayCC = float(str(scale_entryy_2.get()))
            Constant_x = aaxCC/aaxC
            Constant_y = aayCC/aayC
            msg.showinfo("확인하세요", "스케일이 설정되었습니다.")
    print(Constant_x,Constant_y)

    


################### 여기까지 포인팅 ###############
############ 메인 시작 ####################

win = tk.Tk()

win.title("Video Analyzer")

tabcontrol = ttk.Notebook(win)
tab3 = ttk.Frame(tabcontrol)
tabcontrol.add(tab3, text = '이미지 트래킹')
tab4 = ttk.Frame(tabcontrol)
tabcontrol.add(tab4, text="포인팅")
tab2 = ttk.Frame(tabcontrol)
tabcontrol.add(tab2, text="분석 결과")
tabcontrol.pack(expand=1, fill="both")


###########이미지트래킹#############


open_setting=ttk.LabelFrame(tab3)
open_setting.pack(side='top', fill='x')

button_openvideo = ttk.Button(open_setting, text= "Open Video", command=openVideo1)
button_openvideo.pack(side='left',anchor='w')

file__name1 = tk.Label(open_setting, text=filename_1)
file__name1.pack(side='left', padx=30)

button_help = ttk.Button(open_setting, text="도움말", command= __help1)
button_help.pack(side='right')


tabcontrol_1 = ttk.Notebook(tab3)
tab_1 = ttk.Frame(tabcontrol_1)


tabcontrol_1.pack(side='right', anchor='e', fill='y')

tab_1 = ttk.Frame(tabcontrol_1)
tabcontrol_1.add(tab_1, text = '단일 물체 분석')

tab_2 = ttk.Frame(tabcontrol_1)
tabcontrol_1.add(tab_2, text="다중 물체 분석")

objectcho_1 = tk.IntVar()
oneobject = tk.Radiobutton(tab_1, text = "analyze start", value=1, variable=objectcho_1)
oneobject.pack(pady=5)

twoobject = tk.Radiobutton(tab_2, text = "analyze start", value=2, variable=objectcho_1)
twoobject.pack(pady=5)

oneobject.config(value=1, command=open_button)
twoobject.config(value=2, command=open_button)

#========================1차=====

setting_1 = ttk.LabelFrame(tab_1)
setting_1.pack(side='top', fill='y')

aaxframe_1 = tk.Frame(setting_1)
aaxframe_1.pack(side=TOP)
aaxframe_2 = tk.Frame(setting_1)
aaxframe_2.pack(side=TOP)

labelx_1 = tk. Label(aaxframe_1, text= "스케일 상자의 가로축 실제거리 (m) : ")
labelx_1.pack(side=LEFT)

scale_entryx1 = tk.Entry(aaxframe_1, width = 5)
scale_entryx1.insert(0,"0.1")
scale_entryx1.pack(side= LEFT,padx=5)

labely_1 = tk. Label(aaxframe_2, text= "스케일 상자의 세로축 실제거리 (m) : ")
labely_1.pack(side=LEFT)

scale_entryy1 = tk.Entry(aaxframe_2, width = 5)
scale_entryy1.insert(0,"0.1")
scale_entryy1.pack(side= LEFT,padx=5)



set_scale=tk.Button(setting_1, text="set scale",command=s_scale,state=tk.DISABLED)
set_scale.pack( pady=25)

set_Origin = tk.Button(setting_1, text= "set object", command=selectobj,state=tk.DISABLED)
set_Origin.pack(pady=25)


set_start=tk.Button(setting_1, text="start tracking", command=play_video,state=tk.DISABLED)
set_start.pack( pady=25)

reset=tk.Button(setting_1,text='Remove data',command=reset)
reset.pack(pady=25)

datalabel = tk.Label(setting_1, text= "t [ms]  x[pixel]  y[pixel]")
datalabel.pack()

data_box_1 = tk.Frame(setting_1)
data_box_1.pack(side=BOTTOM)

databox_1= tk.Listbox(data_box_1)
databox_1.pack(side=LEFT)

scrollbar_ = tk.Scrollbar(data_box_1,orient="vertical")
scrollbar_.config(command=databox_1.yview)
scrollbar_.pack(side=RIGHT, fill="y")
#===============2차=================================

setting_2 = ttk.LabelFrame(tab_2)
setting_2.pack(side='right', anchor='e', fill='y')



aaxframe_3 = tk.Frame(setting_2)
aaxframe_3.pack(side=TOP,pady=10)
aaxframe_4 = tk.Frame(setting_2)
aaxframe_4.pack(side=TOP)

labelx_4 = tk. Label(aaxframe_3, text= "스케일 상자의 가로축 실제거리 (m) : ")
labelx_4.pack(side=LEFT)

scale_entryx2 = tk.Entry(aaxframe_3, width = 5)
scale_entryx2.insert(0,"0.1")
scale_entryx2.pack(side= LEFT,padx=5)

labely_4 = tk. Label(aaxframe_4, text= "스케일 상자의 세로축 실제거리 (m) : ")
labely_4.pack(side=LEFT)

scale_entryy2 = tk.Entry(aaxframe_4, width = 5)
scale_entryy2.insert(0,"0.1")
scale_entryy2.pack(side= LEFT,padx=5)

set_scale1=tk.Button(setting_2, text="set scale",command=s_scale,state=tk.DISABLED)
set_scale1.pack()


objectcho_2 = tk.IntVar()
oneobject1 = tk.Radiobutton(setting_2, text = "1st object", value=1, variable=objectcho_2)
oneobject1.select()
oneobject1.pack(pady=10)

oneobject2 = tk.Radiobutton(setting_2, text = "2nd object", value=2, variable=objectcho_2)
oneobject2.pack(pady=10)


oneobject1.config(value=1, command=second)
oneobject2.config(value=2, command=second)

set_Origin1 = tk.Button(setting_2, text= "set object", command=selectobj,state=tk.DISABLED)
set_Origin1.pack(pady=25)


set_start1=tk.Button(setting_2, text="start video", command=play_video,state=tk.DISABLED)
set_start1.pack( pady=25)

reset1=tk.Button(setting_2,text='Remove data',command=reset)
reset1.pack(pady=25)

dataframe=tk.Frame(setting_2)
dataframe.pack(side=TOP)

data_box_2 = tk.Frame(dataframe)
data_box_2.pack(side=LEFT)

datalabel = tk.Label(data_box_2, text= "t [ms]  x[pixel]  y[pixel]")
datalabel.pack()

databox_2= tk.Listbox(data_box_2)
databox_2.pack(side=LEFT)

scrollbar_ = tk.Scrollbar(data_box_2,orient="vertical")
scrollbar_.config(command=databox_2.yview)
scrollbar_.pack(side=RIGHT, fill="y")

data_box_3= tk.Frame(dataframe)
data_box_3.pack(side=LEFT)

datalabel = tk.Label(data_box_3, text= "t [ms]  x[pixel]  y[pixel]")
datalabel.pack()

databox_3= tk.Listbox(data_box_3)
databox_3.pack(side=LEFT)

scrollbar_ = tk.Scrollbar(data_box_3,orient="vertical")
scrollbar_.config(command=databox_3.yview)
scrollbar_.pack(side=RIGHT, fill="y")


#================컨버스====
eVideo = tk.Canvas(tab3)
eVideo.pack( fill='both', padx=40 , pady=40)





##########여기까지 이미지트래킹##################

###############포인팅방식####################

move12 = ttk.LabelFrame(tab2)
move12.pack(side= 'left')

move123 = ttk.LabelFrame(move12)
move123.pack()

move1234 = ttk.LabelFrame(move12)
move1234.pack()

open_setting_1=ttk.LabelFrame(tab4)
open_setting_1.pack(side='top', fill='x')

button_openvideo = ttk.Button(open_setting_1, text= "Open Video", command=openVideo2)
button_openvideo.pack(side=LEFT)

file__name2 = tk.Label(open_setting_1, text= fileName)
file__name2.pack(side=LEFT, padx=30)

button_help = ttk.Button(open_setting_1, text="도움말", command= __help2)
button_help.pack(side='right')



packing = tk.Frame(tab4)
packing.pack(side='right', fill='y', anchor='w')

Video = tk.Canvas(tab4)
Video.pack( fill='both', padx=40 , pady=40)
himg=Video.create_image(0,0, image=None, anchor='nw')
Video.bind('<Button-1>', click)
Video.bind('<Button-3>', scaleclick)

tabcontrol121 = ttk.Notebook(packing)
tab11 = ttk.Frame(tabcontrol121)
tabcontrol121.add(tab11, text= "단일 물체 분석")
tab12 = ttk.Frame(tabcontrol121)
tabcontrol121.add(tab12, text= "다중 물체 분석")
tabcontrol121.pack(expand=1, fill="both")



objectcho = tk.IntVar()
oneobject = tk.Radiobutton(tab11, text = "단일물체 분석 활성화", value=1, variable=objectcho)
oneobject.pack(pady=5)
twoobject = tk.Radiobutton(tab12, text = "다중물체 분석 활성화", value=2, variable=objectcho)
twoobject.pack(pady=5)

##첫번째 물체 분석

Method_select = ttk.LabelFrame(tab11, text= "프레임 조절")
Method_select.pack()

check_auto = tk.IntVar()
autocheck = tk.Radiobutton(Method_select, text= "Auto Next", value= 1 ,variable=check_auto)
autocheck.select()
autocheck.pack(pady=5)

manual_frame = ttk.LabelFrame(Method_select)
manual_frame.pack()

active_manual = tk.Radiobutton(manual_frame, text = "Manually next", value= 2, variable=check_auto)
active_manual.pack()

button_nextframe = ttk.Button(manual_frame, text="next Frame", command= nextFrame)
button_nextframe.pack(pady=5)

button_prevframe = ttk.Button(manual_frame, text="prev Frame", command=prevFrame)
button_prevframe.pack(pady=5)

def active():
    if check_auto.get() == 1:
        button_nextframe['state'] = tk.DISABLED
        button_prevframe['state'] = tk.DISABLED
    elif check_auto.get() ==2:
        button_nextframe['state'] = tk.NORMAL
        button_prevframe['state'] = tk.NORMAL

button_removedata = ttk.Button(Method_select, text= "Remove data", command= removedata_1)
button_removedata.pack(pady=5)

autocheck.config(value=1, command=active)
active_manual.config(value=2, command=active)

button_Originpont = ttk.Button(Method_select, text="set scale", command=setscalebtn)
button_Originpont.pack(pady=5)


Entryframe = tk.Frame(Method_select)
Entryframe.pack(pady=5)

aaxframe = tk.Frame(Entryframe)
aaxframe.pack(side=TOP)

aayframe = tk.Frame(Entryframe)
aayframe.pack(side=TOP)

labelx = tk. Label(aaxframe, text= "스케일 상자의 가로축 실제거리 (m) : ")
labelx.pack(side=LEFT)

scale_entryx = tk.Entry(aaxframe, width = 5)
scale_entryx.insert(0,"0.1")
scale_entryx.pack(side= LEFT,padx=5)

labely = tk. Label(aayframe, text= "스케일 상자의 세로축 실제거리 (m) : ")
labely.pack(side=LEFT)

scale_entryy = tk.Entry(aayframe, width = 5)
scale_entryy.insert(0,"0.1")
scale_entryy.pack(side= LEFT,padx=5)



frame_ = tk.Frame(Method_select)
frame_.pack()

frame_2 = tk.Frame(frame_)
frame_2.pack(side='top')


frame_1 = tk.Frame(frame_)
frame_1.pack(side='top')

datalabel = tk.Label(frame_1, text= "물체의 운동")
datalabel.pack()

datalabel = tk.Label(frame_1, text= "t [ms]  x[pixel]  y[pixel]")
datalabel.pack()

data = tk.Listbox(frame_1)
data.pack(side=LEFT)

scrollbar_ = tk.Scrollbar(frame_1,orient="vertical")
scrollbar_.config(command=data.yview)
scrollbar_.pack(side=RIGHT, fill="y")

data.config(yscrollcommand=scrollbar_.set)

scalelabel = tk.Label(frame_2, text= "스케일 측정")
scalelabel.pack()

scalelabel = tk.Label(frame_2, text= " x[pixel]  y[pixel]")
scalelabel.pack()

##진짜 스케일데이터,안보여줌##
scaledata =tk.Listbox(frame_2)

scaledata_show = tk.Listbox(frame_2, height= 5)
scaledata_show.pack(side=BOTTOM)

##두번째 물체 분석
Method_select_2 = ttk.LabelFrame(tab12, text= "프레임 조절")
Method_select_2.pack()

Check_auto_2 = tk.IntVar()
autocheck_2 = tk.Radiobutton(Method_select_2,  value= 1 , text= "Auto Next", variable=Check_auto_2)
autocheck_2.select()
autocheck_2.pack(pady=5)

manual_frame_2 = ttk.LabelFrame(Method_select_2)
manual_frame_2.pack()

active_manual_2 = tk.Radiobutton(manual_frame_2, text = "Manually next", value= 2, variable=Check_auto_2)
active_manual_2.pack()

button_nextframe2 = ttk.Button(manual_frame_2, text="next Frame", command= nextFrame)
button_nextframe2.pack(pady=5)

button_prevframe2 = ttk.Button(manual_frame_2, text="prev Frame", command=prevFrame)
button_prevframe2.pack(pady=5)

def active_2():
    if Check_auto_2.get() == 1:
        button_nextframe2['state'] = tk.DISABLED
        button_prevframe2['state'] = tk.DISABLED
    elif Check_auto_2.get() ==2:
        button_nextframe2['state'] = tk.NORMAL
        button_prevframe2['state'] = tk.NORMAL

autocheck_2.config(value=1, command=active_2)
active_manual_2.config(value=2, command=active_2)

button_removedata_2 = ttk.Button(Method_select_2, text= "Remove data", command=removedata_3)
button_removedata_2.pack(pady=5)

button_Originpont2 = ttk.Button(Method_select_2, text="set scale", command=setscalebtn)
button_Originpont2.pack(pady=5)

nobframe = tk.Frame(Method_select_2)
nobframe.pack(pady=5)

nextobject_2 = tk.IntVar()
nextobject_21 = tk.Radiobutton(nobframe, text= "1st object", var=nextobject_2, value= 1)
nextobject_21.select()
nextobject_21.pack()
nextobject_22 = tk.Radiobutton(nobframe, text= "2nd object", var=nextobject_2, value= 2)
nextobject_22.pack()

def reset__():
    global photo, frameI
    if frames:
        frameI = 0
        if frameI < 0: frameI=0
        photo=frames[frameI][0]
        Video.itemconfig(himg, image=photo)
        hT.config(text=f'{frames[frameI][1]} ms')
    if nextobject_2.get() == 1:
        data_2.delete(0,'end')
    elif nextobject_2.get() ==2:
        data_next_2.delete(0,'end')

nextobject_21.config(value=1, command=reset__)
nextobject_22.config(value=2, command=reset__)

signal = ttk.LabelFrame(move12)
signal.pack()

whatislove = tk.IntVar()
dataploting = tk.Radiobutton(signal, text = "data plot", variable= whatislove, value=1)
dataploting.pack()
datafitting = tk.Radiobutton(signal, text = "data fit", variable= whatislove, value=2)
datafitting.pack()
data123 = tk.Radiobutton(signal, text = "data plot + fit", variable= whatislove, value=3)
data123.pack()


button_analyze_2 = ttk.Button(move12, text= "분석시작", command = analyzebtn)
button_analyze_2.pack(pady=5)

button_remove_2 = ttk.Button(move12, text= "분석결과 초기화", command = removegraph)
button_remove_2.pack(pady=5)


Entryframe_2 = tk.Frame(Method_select_2)
Entryframe_2.pack(pady=5)

aaxframe_2 = tk.Frame(Entryframe_2)
aaxframe_2.pack(side=TOP)

aayframe_2 = tk.Frame(Entryframe_2)
aayframe_2.pack(side=TOP)

labelx_2 = tk. Label(aaxframe_2, text= "스케일 상자의 가로축 실제거리 (m) : ")
labelx_2.pack(side=LEFT)

scale_entryx_2 = tk.Entry(aaxframe_2, width = 5)
scale_entryx_2.insert(0,"0.1")
scale_entryx_2.pack(side= LEFT,padx=5)

labely_2 = tk. Label(aayframe_2, text= "스케일 상자의 세로축 실제거리 (m) : ")
labely_2.pack(side=LEFT)

scale_entryy_2 = tk.Entry(aayframe_2, width = 5)
scale_entryy_2.insert(0,"0.1")
scale_entryy_2.pack(side= LEFT,padx=5)

Radio_1_2=tk.IntVar()

radio1_2=tk.Radiobutton(move123, text="t-x", value=1, variable=Radio_1_2)
radio1_2.pack(pady=5)

radio2_2=tk.Radiobutton(move123, text="t-y", value=2, variable=Radio_1_2)
radio2_2.pack(pady=5)

radio3_2=tk.Radiobutton(move123, text="x-y", value=3, variable=Radio_1_2)
radio3_2.pack(pady=5)


frame__2 = tk.Frame(Method_select_2)
frame__2.pack()

frame_2_2 = tk.Frame(frame__2)
frame_2_2.pack(side='top')


frame_1_2 = tk.Frame(frame__2)
frame_1_2.pack(side='top')

datalabel_2 = tk.Label(frame_1_2, text= "물체의 운동")
datalabel_2.pack()

datalabel_2 = tk.Label(frame_1_2, text= "t [ms]  x[pixel]  y[pixel]")
datalabel_2.pack()

data_2 = tk.Listbox(frame_1_2)
data_2.pack(side=LEFT)

data_next_2 = tk.Listbox(frame_1_2)
data_next_2.pack(side= 'left')

scrollbar__2 = tk.Scrollbar(frame_1_2,orient="vertical")
scrollbar__2.config(command=data_2.yview)
scrollbar__2.pack(side=RIGHT, fill="y")

data_2.config(yscrollcommand=scrollbar__2.set)

scalelabel_2 = tk.Label(frame_2_2, text= "스케일 측정")
scalelabel_2.pack()

scalelabel_2 = tk.Label(frame_2_2, text= " x[pixel]  y[pixel]")
scalelabel_2.pack()

##진짜 스케일데이터,안보여줌##
scaledata_2 =tk.Listbox(frame_2_2)

scaledata_show_2 = tk.Listbox(frame_2_2, height= 5)
scaledata_show_2.pack(side=BOTTOM)



hT=tk.Label(Method_select_2, text='0 ms') 
hSize=tk.Label(tab4, text='0x0')



###########여기까지 포인팅방식#################

win.mainloop()


