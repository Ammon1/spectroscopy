# Simple enough, just import everything from tkinter.
import gc
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import pandas as pd

import numpy as np
import matplotlib
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(tk.Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        tk.Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget      
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=tk.BOTH, expand=1)

        # creating a menu instance
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        # create the file object)
        file = tk.Menu(menu)
        #opening file
        file.add_command(label = 'Open', command = self.OpenFile)
        #open reference spectrum
        file.add_command(label = 'Open ref', command = self.OpenRef)
        #cleaning the data
        file.add_command(label = 'Clean data', command = self.CleanData)
        #subtract substrate
        file.add_command(label = 'Subtract substrate', command = self.SubtSubst)
        
        file.add_command(label = 'Rescale ref', command = self.RescaleRef)
        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        #added "file" to our menu
        menu.add_cascade(label="File", menu=file)

        # create the file object)
        edit = tk.Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        edit.add_command(label="Undo")

        #added "file" to our menu
        menu.add_cascade(label="Edit", menu=edit)
        
        self.var = tk.StringVar()
        self.ref_var = tk.DoubleVar()
        self.data_var = tk.DoubleVar()
        
        label = tk.Label( self.master, textvariable=self.var, relief=tk.RAISED )
        label.pack()
        
        
        self.PlotData(0,'')
        self.PlotRef(0,'')
        #comment ofcurrent stage
        self.var.set("Begining")

    #closing program
    def client_exit(self):
        self.master.destroy()
    
    #main operation program
    def OpenFile(self):
        self.var.set("Opening file")
        
        name = askopenfilename(initialdir="C:/Users/",
                           filetypes =(("Text File", "*.txt"),("All Files","*.*")),
                           title = "Choose a file."
                           )
        df=pd.read_csv(name,names=['x','y','lambda','int'],sep='\t')
        table=df.iloc[:,3].values
        
        self.lambdas,self.rows,index_num=self.GetLambdas(df)
        
        intensities=np.reshape(table,(self.rows,index_num))
        intensities=np.transpose(intensities)
        self.intensities=intensities

        self.df_int=pd.DataFrame(self.intensities)
        self.var.set("data frame constructed")
        
        x=self.lambdas[:1011]
        y=self.intensities[:,0]
        
        self.lineD.set_ydata(y)
        self.lineD.set_xdata(x)
        
        ax = self.canvas_data.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())  
    
        self.canvas_data.draw()
        
        
        #self.data_var=intensities[:,0]
        #self.FigData=a.plot(self.data_var)
        #self.canvas_data.draw()
        
        #f = Figure(figsize=(5,2), dpi=100)
        #a = f.add_subplot(111)
        #a.plot(intensities[:,0])
        #
        #canvas = FigureCanvasTkAgg(f, self)
        #canvas.draw()
        #canvas.get_tk_widget().pack(side=tk.TOP, expand=False)
        
    #bardziej skomplikowane- trzeba uwzględnićczy jest to mapa czy pojedyncze spektrum
    def OpenRef(self):
        self.var.set("Opening ref")
        
        name = askopenfilename(initialdir="C:/Users/",
                           filetypes =(("Text File", "*.txt"),("All Files","*.*")),
                           title = "Choose a file."
                           )
        df_ref=pd.read_csv(name,names=['x','y','lambda','int'],sep='\t')
        table_ref=df_ref.iloc[:,3].values
        
        self.lambdas,self.rows,index_num=self.GetLambdas(df_ref)
        
        int_ref=table_ref[:index_num]
        
        self.int_ref=int_ref-int_ref[0]
        #normalization of ref to spectrum
        self.var.set("ref data frame constructed")
        
        #self.PlotRef(self.int_ref,'ref')
       
        #self.data_var=self.intensities_new[:,0]
        #self.canvas_data.draw()
        
        x=self.lambdas[:1011]
        y=self.int_ref[:]
        
        self.lineR.set_ydata(y)
        self.lineR.set_xdata(x)
        
        ax = self.canvas_ref.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())  
    
        self.canvas_ref.draw()
        #f = Figure(figsize=(5,2), dpi=100)
        #a = f.add_subplot(111)
        #a.plot(self.int_ref)
        
        #canvas = FigureCanvasTkAgg(f, self)
        #canvas.draw()
        #canvas.get_tk_widget().pack(side=tk.BOTTOM, expand=False)

        #toolbar = NavigationToolbar2TkAgg(canvas, self)
        #toolbar.update()
        #canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def RescaleRef(self):
        self.var.set("rescale ref to map")

        int_ref=self.int_ref
        #ref_max=np.argmax(int_ref)
        try:
            intensities= self.intensities
        except:
            self.var.set("failed!")
        
        rescaler=(intensities[800,0]/int_ref[800])
        
        print(int_ref[:20])
        self.int_ref=int_ref*rescaler
        print(rescaler)
        print(int_ref[:20])
        
        x=self.lambdas[:1011]
        y=self.int_ref[:]
        
        self.lineR.set_ydata(y)
        self.lineR.set_xdata(x)
            
        ax = self.canvas_ref.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())  
        
        self.canvas_ref.draw()
    
          
        
        
    def SubtSubst(self):
        self.var.set("subtracting substrate")
        
        ref=self.int_ref[10:-10]
        intensities=self.intensities
        intensities_sub=np.zeros(intensities.shape)
        mem=0
        suma=1e12
        for j in range(0,intensities.shape[1]):
            for i in range (0,300,1):
                result=intensities[:,0]-0.1*i*ref
                if suma>(sum(abs(result[600:900]-result[600]))):
                    suma=sum(abs(result[600:900]-result[600]))
                    mem=i
            intensities_sub[:,j]=intensities[:,j]-0.1*mem*ref
        
        #nie plot tylko update
        x=self.lambdas[:intensities_sub.shape[0]]
        y=intensities_sub[:,0]
        
        self.lineD.set_ydata(y)
        self.lineD.set_xdata(x)
        
        ax = self.canvas_data.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())  
    
        self.canvas_data.draw()
        #self.PlotFigure(intensities_sub[:,0],'subtracted data')
        
    def CleanData(self):
        self.var.set("data cleaning")
        df=self.df_int
        
        df_roll_20=df.rolling(20).mean().shift(-10)
        df_roll_4=df.rolling(4).mean().shift(-2)
        df_clear=np.abs(df-df_roll_4)/np.sqrt(np.abs(df_roll_20))

        df_clear=df_clear.fillna(df_clear.mean())
        df_spike=df_clear>7

        df_anti_spike=~df_spike.astype(int)+2
        df_spike=df_spike.astype(int)
        df_new=df_spike*df_roll_20+df_anti_spike*df
        df_new=df_new.iloc[10:-10,:]
        
        self.df_cleaned=df_new
        print(self.df_cleaned.shape,'\n',self.df_int.shape)
        self.var.set("spikes removed")
        
        self.var.set("scaling data")
        scaler = StandardScaler()
        intensities=scaler.fit_transform(df_new.loc[:,:].values)
        transformer = FastICA(n_components=10,random_state=0)
        X_transformed = transformer.fit_transform(intensities)
        #helped empty matrix
        self.intensities_new=np.zeros(intensities.shape)
        
        self.var.set("constructing data from ICA vectors")
        mem=-1
        #for (x,y),value in c(np.transpose(intensities)):
        for x in range(0,intensities.shape[1]):
            if x!=mem:
                popt, pcov = curve_fit(self.grafen_plot, X_transformed, intensities[:,x])
                self.intensities_new[:,x]=self.grafen_plot(X_transformed, *popt)
                mem=x
        
        x=self.lambdas[:self.intensities_new.shape[0]]
        y=self.intensities_new[:,0]
        
        self.lineD.set_ydata(y)
        self.lineD.set_xdata(x)
        
        ax = self.canvas_data.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())  
    
        self.canvas_data.draw()
        
        self.popupmsg()
        
        #self.PlotData(self.intensities_new[:,0],'data cleaned')
        self.var.set("data cleaning succed!")
        
    #function to get x-axis from data
    def GetLambdas(self,data):
        self.var.set("calculating lambdas")

        for index, row in data.iterrows():
            if index==0:
                lambda_num=row['lambda']
            elif row['lambda']==lambda_num and index!=0:
                    index_num=index
                    break
        
        return data.loc[:index_num,'lambda'].values,int(data.shape[0]/index_num),index_num
    
    def grafen_plot(self,X,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9):
        y=X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3+X[:,4]*a4+X[:,5]*a5
        y+=X[:,6]*a6+X[:,7]*a7+X[:,8]*a8+X[:,9]*a9
        return y
    
    def PlotData(self,data,text):
        
        fD = Figure(figsize=(5,2), dpi=100)
        a = fD.add_subplot(111)
        self.data_var=data
        self.lineD,=a.plot(self.data_var)
        a.set_title(text)
        
        self.canvas_data = FigureCanvasTkAgg(fD, self)
        self.canvas_data.draw()
        self.canvas_data.get_tk_widget().pack(side=tk.TOP, expand=False)
        
    def PlotRef(self,ref,text):
        
        fR = Figure(figsize=(5,2), dpi=100)
        a = fR.add_subplot(111)
        self.ref_var=ref
        self.lineR,=a.plot(self.ref_var)
        a.set_title(text)
        
        self.canvas_ref = FigureCanvasTkAgg(fR, self)
        self.canvas_ref.draw()
        self.canvas_ref.get_tk_widget().pack(side=tk.BOTTOM, expand=False)
        
    def popupmsg(self):
        NORM_FONT= ("Verdana", 10)
        self.popup = tk.Tk()
        self.popup.geometry("200x200")
        self.popup.wm_title("!")
        label = ttk.Label(self.popup, text="do You accept?", font=NORM_FONT)
        label.pack(side="top")
        B1 = ttk.Button(self.popup, text="Yes", command = self.PopYes)
        B2 = ttk.Button(self.popup, text="No", command = self.PopNo)
        B1.pack()
        B2.pack()
        #popup.mainloop()
        
    def PopYes(self):
        self.popup.destroy()
        self.intensities=self.intensities_new
        del self.intensities_new
        gc.collect()
        
        x=self.lambdas[:self.intensities.shape[0]]
        y=self.intensities[:,0]
        
        self.lineD.set_ydata(y)
        self.lineD.set_xdata(x)
        
        ax = self.canvas_data.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())  
    
        self.canvas_data.draw()
        
    def PopNo(self):
        self.popup.destroy()
        del self.intensities_new
        gc.collect()
        
        x=self.lambdas[:self.intensities.shape[0]]
        y=self.intensities[:,0]
        
        self.lineD.set_ydata(y)
        self.lineD.set_xdata(x)
        
        ax = self.canvas_data.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())  
    
        self.canvas_data.draw()
        
        

        
# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = tk.Tk()

root.geometry("700x700")

#creation of an instance
app = Window(master=root)

#mainloop 
root.mainloop()  