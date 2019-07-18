import gc
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import pandas as pd

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from matplotlib.figure import Figure

import time




# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(tk.Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)                    
        self.master = master
        self.intensities=np.zeros(3)
        self.lambdas=np.zeros(3)
        self.progressbar = ttk.Progressbar(self.master, maximum = 100,  orient='horizontal', mode='determinate')
        
        self.init_window()
    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget      
        self.master.title("Renishaw")
        self.pack(fill=tk.BOTH, expand=1)

        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        # create the file object)
        file = tk.Menu(menu)
        #opening file
        file.add_command(label = 'Open', command = self.open_file)
        #open reference spectrum
        file.add_command(label = 'Open ref', command = self.open_ref)
        #cleaning the data
        file.add_command(label = 'Clean data', command = self.clean_data)
        #subtract substrate
        file.add_command(label = 'Subtract substrate', command = self.subt_subst)
        
        file.add_command(label = 'Rescale ref', command = self.rescale_ref)
        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        #added "file" to our menu
        menu.add_cascade(label="File", menu=file)
        
        fit = tk.Menu(menu)
        fit.add_command(label = '2D', command = self.fit2D)
        fit.add_command(label = 'G', command = self.fitG)
        fit.add_command(label = 'D', command = self.fitD)
        fit.add_command(label = 'Hydrogen', command = self.fitH2)
        menu.add_cascade(label="Fitting", menu=fit)
        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
     
        self.progressbar["value"] = 0
        self.progressbar.pack(expand=True, fill=tk.BOTH, side=tk.TOP)
        
        self.var = tk.StringVar()
        self.ref_var = tk.DoubleVar()
        self.data_var = tk.DoubleVar()
        
        label = tk.Label( self.master, textvariable=self.var, relief=tk.RAISED )
        label.pack()
        
        
        self.plot_data(0,'')
        self.plot_ref(0,'')
        #comment ofcurrent stage
        self.var.set("Begining")
        

    #closing program
    def client_exit(self):
        self.master.destroy()
    
    #main operation program
    def open_file(self):
        self.var.set("Opening file")
        
        self.intensities,self.lambdas,self.pos_x,self.pos_y,self.rows=open_file()
        
        x=self.lambdas[:1011]
        y=self.intensities[:,0]
        
        self.refres_draw(x,y,self.lineD,self.canvas_data)
        
        
        
    def open_ref(self):
        self.var.set("Opening ref")

        self.int_ref,lambdas=open_ref()
        
        self.var.set("ref data frame constructed")
        
        y=self.int_ref[:]
        x=lambdas[:y.shape[0]]
        
        self.refres_draw(x,y,self.lineR,self.canvas_ref)
        
        
    def rescale_ref(self):
        self.var.set("rescale ref to map")

        int_ref=self.int_ref
        try:
            
            rescaler=(self.intensities[800,0]/int_ref[800])
        except:
            self.var.set("failed!")
        
        self.int_ref=int_ref*rescaler
        
        length=np.minimum(self.int_ref.shape[0],self.lambdas.shape[0])
        y=self.int_ref[:length]
        x=self.lambdas[:length]
        
        self.refres_draw(x,y,self.lineR,self.canvas_ref)
     
    def subt_subst(self):
        
        self.var.set("subtracting substrate")
        
        self.intensities_new=subt_subst(self.int_ref,self.intensities,self.master,self.progressbar,self.var)
        x=self.lambdas[:self.intensities_new.shape[0]]
        y=self.intensities_new[:,0]
        
        self.refres_draw(x,y,self.lineD,self.canvas_data)
 
        self.popupmsg()
        
    def clean_data(self):
        self.intensities_new=clean_data(self.intensities,self.master,self.progressbar,self.var)
        
        x=self.lambdas[:self.intensities_new.shape[0]]
        y=self.intensities_new[:,0]
        
        self.refres_draw(x,y,self.lineD,self.canvas_data)

        self.popupmsg()
        
        self.var.set("data cleaning succed!")
        
    def plot_data(self,data,text):
        
        fD = Figure(figsize=(5,2), dpi=100)
        a = fD.add_subplot(111)
        self.data_var=data
        self.lineD,=a.plot(self.data_var)
        a.set_title(text)
        
        self.canvas_data = FigureCanvasTkAgg(fD, self)
        self.canvas_data.draw()
        self.canvas_data.get_tk_widget().pack(side=tk.TOP, expand=False)
        
        toolbar = NavigationToolbar2Tk(self.canvas_data, self)
        toolbar.update()
        self.canvas_data.get_tk_widget().pack(side=tk.BOTTOM, expand=False)
        
        
    def plot_ref(self,ref,text):
        
        fR = Figure(figsize=(5,2), dpi=100)
        a = fR.add_subplot(111)
        self.ref_var=ref
        self.lineR,=a.plot(self.ref_var)
        a.set_title(text)
        
        self.canvas_ref = FigureCanvasTkAgg(fR, self)
        self.canvas_ref.draw()
        self.canvas_ref.get_tk_widget().pack(side=tk.BOTTOM, expand=False)
        
        toolbar = NavigationToolbar2Tk(self.canvas_ref, self)
        toolbar.update()
        self.canvas_ref.get_tk_widget().pack(side=tk.TOP, expand=False)
        
    def popupmsg(self):
        self.popup = tk.Tk()
        self.popup.geometry("200x200")
        self.popup.wm_title("!")
        label = ttk.Label(self.popup, text="do You accept?")
        label.pack(side="top")
        B1 = ttk.Button(self.popup, text="Yes", command = self.pop_yes)
        B2 = ttk.Button(self.popup, text="No", command = self.pop_no)
        B1.pack()
        B2.pack()
        #popup.mainloop()
        
    def pop_yes(self):
        self.popup.destroy()
        self.intensities=self.intensities_new
        
        del self.intensities_new
        gc.collect()
        self.lambdas=self.lambdas[:self.intensities.shape[0]]
        
        length=np.minimum(self.intensities.shape[0],self.lambdas.shape[0])
        x=self.lambdas[:length]
        y=self.intensities[:length,0]
        
        self.refres_draw(x,y,self.lineD,self.canvas_data)
        
    def pop_no(self):
        self.popup.destroy()
        del self.intensities_new
        gc.collect()
        length=np.minimum(self.intensities.shape[0],self.lambdas.shape[0])
        x=self.lambdas[:length]
        y=self.intensities[:length,0]
        
        self.refres_draw(x,y,self.lineD,self.canvas_data)
    
    def refres_draw(self,x,y,line,canvas):
        
        line.set_ydata(y)
        line.set_xdata(x)
        
        ax = canvas.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())  
    
        canvas.draw()
    
    def fit2D(self):
        poz2D=2680
        FWHM2D=30
        
        map_pos2D=np.empty(1)
        map_FWHM2D=np.empty(1)
        
        index=(np.abs(self.lambdas - poz2D)).argmin()
        
        
        
        ax,line2D,line2D_fit,canvas2=draw_fitting('fit2D',self.lambdas,self.intensities[:,0])
        
        for i in range (0,self.intensities.shape[1]):
          
            x=self.lambdas[index-100:index+100]
            y=self.intensities[index-100:index+100,i]
            max2D=np.max(y)
            poz2D=x[np.where(y==np.max(y))]
            
            popt_saved=[self.intensities[0,i],max2D,poz2D,FWHM2D]
            
            try:
                popt, pcov = curve_fit(fit_lorenz, x , y,p0=popt_saved,
                                   bounds=([np.min(y),0,poz2D-40,FWHM2D-15],[np.max(y),100*max2D,poz2D+60,FWHM2D+40]))
                
            except:
                popt=[np.min(y),0,poz2D,30]
                print('failed ')
            
            line2D.set_ydata(y)
            line2D.set_xdata(x)
            line2D_fit.set_ydata(fit_lorenz(x, *popt))
                
            line2D_fit.set_xdata(x)
            try:
                ax.set_ylim(np.min(y),np.max(y))
                canvas2.draw()
                
            except:
                print('failed to draw')
            
            self.progressbar["value"] = (i/self.intensities.shape[1])*100
            self.progressbar.update()
            par=[]
            for o in popt:
                par.append(int(o))
            self.var.set(str(int(i/self.intensities.shape[1]*100)) + '%'+str(par))
            
            map_pos2D=np.append(map_pos2D,popt[3])
            map_FWHM2D=np.append(map_FWHM2D,popt[2])
        
        map_FWHM2D=map_FWHM2D[1:].reshape(self.pos_x.shape[0],self.pos_y.shape[0])
        map_pos2D=map_pos2D[1:].reshape(self.pos_x.shape[0],self.pos_y.shape[0])
        
        draw_maps(map_FWHM2D,'FWHM 2D map',map_pos2D,'position 2D map',self.pos_x,self.pos_y)
        
        
        
    def fitG(self):
        pozG=1580
        FWHMG=15
        map_posG=np.empty(1)
        map_FWHMG=np.empty(1)
        
        index=(np.abs(self.lambdas - pozG)).argmin()
        
        
        ax,lineG,lineG_fit,canvas2=draw_fitting('fitG',self.lambdas,self.intensities[:,0])
        
        for i in range (0,self.intensities.shape[1]):
            
            x=self.lambdas[index-100:index+100]
            y=self.intensities[index-100:index+100,i]
            maxG=np.max(y)
            pozG=x[np.where(y==np.max(y))]
            popt_saved=[self.intensities[0,i],maxG,pozG,FWHMG]
            try:
                popt, pcov = curve_fit(fit_lorenz, x , y,p0=popt_saved,
                                   bounds=([np.min(y),0,pozG-10,FWHMG-5],[np.max(y),10*maxG,pozG+60,FWHMG+20]))
                
            except:
                popt=[0,0,1580,15]
                print('failed ')
            
            lineG.set_ydata(y)
            lineG.set_xdata(x)
            
            lineG_fit.set_ydata(fit_lorenz(x, *popt))
            lineG_fit.set_xdata(x)
            try:
                ax.set_ylim(np.min(y),np.max(y))
                canvas2.draw()
               
            except:
                print('failed to draw')
            
            self.progressbar["value"] = (i/self.intensities.shape[1])*100
            self.progressbar.update()
            par=[]
            for o in popt:
                par.append(int(o))
            self.var.set(str(int(i/self.intensities.shape[1]*100)) + '%'+str(par))
            
            map_posG=np.append(map_posG,popt[3])
            map_FWHMG=np.append(map_FWHMG,popt[2])
        
        map_FWHMG=map_FWHMG[1:].reshape(self.pos_x.shape[0],self.pos_y.shape[0])
        map_posG=map_posG[1:].reshape(self.pos_x.shape[0],self.pos_y.shape[0])
        
        draw_maps(map_FWHMG,'FWHM G map',map_posG,'position G map',self.pos_x,self.pos_y)
        
    def fitD(self):
        pozD=1350
        FWHMD=20
        
        index=(np.abs(self.lambdas - pozD)).argmin()
        
        map_posD=np.empty(1)
        map_FWHMD=np.empty(1)
        
        
        
        ax,lineD,lineD_fit,canvas2=draw_fitting('fitD',self.lambdas,self.intensities[:,0])
        
        for i in range (0,self.intensities.shape[1]):
            
            x=self.lambdas[index-50:index+50]
            y=self.intensities[index-50:index+50,i]
            
            pozD=x[np.where(y==np.max(y))]
           
            maxD=np.max(y)
            
            popt_saved=[self.intensities[0,i],maxD,pozD,FWHMD]
            try:
                popt, pcov = curve_fit(fit_lorenz, x , y,p0=popt_saved,
                                   bounds=([0,0,pozD-10,FWHMD-5],[np.max(y),10*maxD,pozD+60,FWHMD+20]))
                
            except:
                popt=[0,0,1350,20]
                print('failed ')
            
            lineD.set_ydata(y)
            lineD.set_xdata(x)
            lineD_fit.set_ydata(fit_lorenz(x, *popt))
            lineD_fit.set_xdata(x)
            try:
                ax.set_ylim(np.min(y),np.max(y))
                canvas2.draw()
            except:
                print('failed to draw')
            
            self.progressbar["value"] = (i/self.intensities.shape[1])*100
            self.progressbar.update()
            par=[]
            for o in popt:
                par.append(int(o))
            self.var.set(str(int(i/self.intensities.shape[1]*100)) + '%'+str(par))
            
            map_posD=np.append(map_posD,popt[3])
            map_FWHMD=np.append(map_FWHMD,popt[2])
        
        map_FWHMD=map_FWHMD[1:].reshape(self.pos_x.shape[0],self.pos_y.shape[0])
        map_posD=map_posD[1:].reshape(self.pos_x.shape[0],self.pos_y.shape[0])
        
        draw_maps(map_FWHMD,'FWHM D map ',map_posD,'position D map',self.pos_x,self.pos_y)
    
    def fitH2(self):
        pozH2=2145
        FWHMH2=1
        index=(np.abs(self.lambdas - pozH2)).argmin()
        
        map_posH2=np.empty(1)
        map_INTH2=np.empty(1)

        popt_saved=[self.intensities[index-10,0],0.1,pozH2,FWHMH2]
        ax,lineH2,lineH2_fit,canvas2=draw_fitting('fitH2',self.lambdas[index-10:index+10],self.intensities[index-10:index+10,0])
        
        for i in range (0,self.intensities.shape[1]):
            
            x=self.lambdas[index-10:index+10]
            y=self.intensities[index-10:index+10,i]
            
            pozH2=x[np.where(y==np.max(y))]
            
            
            intH2=np.max(y)-np.min(y)
            
            bounds=([np.min(y),0,pozH2-10,0],[np.max(y),10*intH2,pozH2+20,FWHMH2+10])
            failed=False
            try:
                popt, pcov = curve_fit(fit_lorenz, x , y,p0=popt_saved,
                                   bounds=bounds) 
            except:
                popt=[np.min(y),intH2,pozH2,FWHMH2]
                failed=True
                print('failed ')
            
            lineH2.set_ydata(y)
            lineH2.set_xdata(x)
            
            lineH2_fit.set_ydata(fit_lorenz(x, *popt))
            lineH2_fit.set_xdata(x)
            ax.set_ylim(np.min(y),np.max(y))
            canvas2.draw()
                
            if failed == True:
                time.sleep(1) 
                print('failed to draw')
            
            self.progressbar["value"] = (i/self.intensities.shape[1])*100
            self.progressbar.update()
            par=[]
            for o in popt:
                par.append(o)
            self.var.set(str(int(i/self.intensities.shape[1]*100)) + '%'+str(par))
            #popt_saved=popt
            
            map_posH2=np.append(map_posH2,popt[3])
            map_INTH2=np.append(map_INTH2,popt[1])
        
        map_INTH2=map_INTH2[1:].reshape(self.pos_x.shape[0],self.pos_y.shape[0])
        map_posH2=map_posH2[1:].reshape(self.pos_x.shape[0],self.pos_y.shape[0])
        
        draw_maps(map_INTH2,'H2 intensity',map_posH2,'H2 position',self.pos_x,self.pos_y)
    
def fit_lorenz(E,y0,I0,E0,gamma):
    return (y0+I0*((gamma**2)/(((E-E0)**2)+gamma**2)))
    
        
root = tk.Tk()

root.geometry("700x600")

#creation of an instance
app = Window(master=root)


def get_lambdas(data):

        for index, row in data.iterrows():
            if index==0:
                lambda_num=row['lambda']
            elif row['lambda']==lambda_num and index!=0:
                    index_num=index
                    break
        
        return data.loc[:index_num,'lambda'].values,int(data.shape[0]/index_num),index_num
    
def open_file():
        name = askopenfilename(initialdir="C:/Users/",
                           filetypes =(("Text File", "*.txt"),("All Files","*.*")),
                           title = "Choose a file.")  
        df=pd.read_csv(name,names=['x','y','lambda','int'],sep='\t')
        table=df.iloc[:,3].values
        #self.lambdas,self.rows,index_num=self.get_lambdas(df)
        lambdas,rows,index_num=get_lambdas(df)
        pos_x=df.iloc[:,0].unique()
        pos_y=df.iloc[:,1].unique()
          
        intensities=np.reshape(table,(rows,index_num))
        intensities=np.transpose(intensities)
        #position verification
        x=0
        y=0
        correct=True
        for pos in range(0,intensities.shape[1]):
            if df.iloc[index_num*pos:index_num*pos+1,0].values == x:
                if y != pos_y[(pos-1)%pos_x.shape[0]]:
                    #print('1 ',x,y,' ',pos_y[pos%pos_x.shape[0]])
                    correct=False
            elif df.iloc[index_num*pos:index_num*pos+1,1].values == y:
                if x != pos_x[(pos-1)%pos_y.shape[0]]:
                    #print('2 ',x,y)
                    correct=False
            x=df.iloc[index_num*pos:index_num*pos+1,0].values
            y=df.iloc[index_num*pos:index_num*pos+1,1].values
         
        if correct == True:
            print('rectangular map')
        else:
            print('unregular map- You wont make maps of points!')
        intensities=intensities
        lambdas=lambdas[:intensities.shape[0]]
        return intensities,lambdas,pos_x,pos_y,rows
    
def open_ref():
        name = askopenfilename(initialdir="C:/Users/",
                           filetypes =(("Text File", "*.txt"),("All Files","*.*")),
                           title = "Choose a file."
                           )
        try:
            df_ref=pd.read_csv(name,sep='\t')
        except:
            print('no file')
        if df_ref.shape[1]==4:
            df_ref=pd.read_csv(name,names=['x','y','lambda','int'],sep='\t')
            table_ref=df_ref.iloc[:,3].values
            lambdas,rows,index_num=get_lambdas(df_ref)
            int_ref=table_ref[:index_num]
            
        elif df_ref.shape[1]==2:
            df_ref=pd.read_csv(name,names=['lambda','int'],sep='\t')
            #table_ref=df_ref.iloc[:,1].values
            lambdas=df_ref.iloc[:,0].values
            #rows=df_ref.shape[0]
            int_ref=df_ref.iloc[:,1].values
            
        else:
            print('wrong format')
        
        int_ref=int_ref-int_ref[0]
        return int_ref,lambdas
    
def subt_subst(int_ref,intensities,master,progressbar,var):
        
        ref=int_ref[10:-10]
        intensities_new=np.zeros(intensities.shape)
        mem=0
        suma=1e12
        for j in range(0,intensities.shape[1]):
            progressbar["value"] = (j/intensities.shape[1])*100
            progressbar.update()
            var.set(str(int(j/intensities.shape[1]*100)) + '%')
            
            for i in range (0,300,1):
                result=intensities[:,0]-0.01*i*ref
                if suma>(sum(abs(result[600:900]-result[600]))):
                    suma=sum(abs(result[600:900]-result[600]))
                    mem=i
            intensities_new[:,j]=intensities[:,j]-0.01*mem*ref
        return intensities_new
    
def clean_data(intensities,master,progressbar,var):
    
        
        var.set("data cleaning")
        df=pd.DataFrame(intensities)
        
        del intensities
        gc.collect()
        
        var.set("constructing aditional df")
        df_roll_20=df.rolling(20).mean().shift(-10)
        df_roll_4=df.rolling(4).mean().shift(-2)
        df_clear=np.abs(df-df_roll_4)/np.sqrt(np.abs(df_roll_20))

        df_clear=df_clear.fillna(df_clear.mean())
        df_spike=df_clear>7

        df_anti_spike=~df_spike.astype(int)+2
        df_spike=df_spike.astype(int)
        df_new=df_spike*df_roll_20+df_anti_spike*df
        df_new=df_new.iloc[10:-10,:]
        
        
        del df_roll_20,df_roll_4,df_clear,df_spike,df_anti_spike,
        
        #self.df_cleaned=df_new
        
        var.set("spikes removed")
        
        var.set("scaling data")
        scaler = StandardScaler()
        intensities=scaler.fit_transform(df_new.loc[:,:].values)
        transformer = FastICA(n_components=10,random_state=0)
        X_transformed = transformer.fit_transform(intensities)
        #helped empty matrix
        
        intensities_new=np.zeros(intensities.shape)
        
        var.set("constructing data from ICA vectors")
        
        for x in range(0,intensities.shape[1]):
            progressbar["value"] = (x/intensities.shape[1])*100
            progressbar.update()
            var.set(str(int(x/intensities.shape[1]*100)) + '%')
            try:
                popt, pcov = curve_fit(grafen_plot, X_transformed, intensities[:,x])
                intensities_new[:,x]=grafen_plot(X_transformed, *popt)
            except:
                print(x,' fail to fit')
                intensities_new[:,x]=intensities[:,x]
        return intensities_new
    
def grafen_plot(X,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9):
        y=X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3+X[:,4]*a4+X[:,5]*a5
        y+=X[:,6]*a6+X[:,7]*a7+X[:,8]*a8+X[:,9]*a9
        return y
    
def draw_fitting(name,lambdas,intensities):
        root2 = tk.Toplevel()
        fig2 = plt.Figure()
        canvas2 = FigureCanvasTkAgg(fig2, master=root2)
        canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1.0)
        ax = fig2.add_subplot(111)
        
        x_draw,y_draw=lambdas,intensities
        x_draw_fit,y_draw_fit=lambdas,intensities
    
        ax.set_title(name)
        line2D,=ax.plot(x_draw,y_draw)
        line2D_fit,=ax.plot(x_draw_fit,y_draw_fit)
        #ax.plot(x_draw,y_draw)
        canvas2.draw()
        return ax,line2D,line2D_fit,canvas2
    
def draw_maps(map_FWHM2D,tite1,map_pos2D,title2,x_ticks,y_ticks):
        root2 = tk.Toplevel()
        root2.geometry("400x600")
        fig3 = plt.Figure()
        canvas3 = FigureCanvasTkAgg(fig3, master=root2)
        canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        ax = fig3.add_subplot(111)
        ax.set_title(tite1)
        ax.matshow(map_FWHM2D)
        fig3.colorbar(ax.matshow(map_FWHM2D))
        ax.set_xticklabels(x_ticks)  
        ax.set_yticklabels(y_ticks)
        
        fig4=plt.Figure()
        canvas3 = FigureCanvasTkAgg(fig4, master=root2)
        canvas3.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)
        ax = fig4.add_subplot(111)
        ax.set_title(title2)
        ax.matshow(map_pos2D)
        fig4.colorbar(ax.matshow(map_pos2D))
        ax.set_xticklabels(x_ticks)  
        ax.set_yticklabels(y_ticks)
        
        canvas3.draw()
#mainloop 
root.mainloop()  