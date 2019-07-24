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


class Window(tk.Frame):

    def __init__(self,master,opening,data_manipulation,fitting,drawing):
        tk.Frame.__init__(self, master)                    
        self.master = master
        self.init_window()
        self.opening=opening
        self.data_manipulation=data_manipulation 
        self.fitting=fitting
        self.drawing=drawing
        
    def init_window(self,):
        # master title     
        self.master.title("Renishaw")
        self.pack(fill=tk.BOTH, expand=1)
        # menu file
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)
        file = tk.Menu(menu)
        file.add_command(label = 'Open data', command = self.open_file)
        file.add_command(label = 'Open reference', command = self.open_reference)
        file.add_command(label = 'Clean data', command = self.clean_data)
        file.add_command(label = 'Subtract substrate', command = self.substrate_subtraction)
        file.add_command(label = 'Rescale reference', command = self.rescale_reference)
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)
        # menu fit
        fit = tk.Menu(menu)
        fit.add_command(label = '2D', command = self.fit2D)
        fit.add_command(label = 'G', command = self.fitG)
        fit.add_command(label = 'D', command = self.fitD)
        fit.add_command(label = 'Si-H', command = self.fitSi_H)
        menu.add_cascade(label="Fitting", menu=fit)
        #progress bar
        self.progressbar = ttk.Progressbar(self.master, maximum = 100,  orient='horizontal', mode='determinate')
        self.progressbar["value"] = 0
        self.progressbar.pack(expand=True, fill=tk.BOTH, side=tk.BOTTOM)
        # status label
        self.status_label = tk.StringVar()
        self.status_label.set("Begining")
        label = tk.Label( self.master, textvariable=self.status_label, relief=tk.RAISED,height=2,width=30 )
        label.pack()
        #reference spectrum
        self.line_data,self.canvas_data=drawing.draw_master(0,'Data',self.master)
        self.line_reference,self.canvas_reference=drawing.draw_master(0,'Reference',self.master)
        
    def client_exit(self):
        self.master.destroy()
    
    def open_file(self):
        self.intensities,self.lambdas,self.position_x,self.position_y,self.rows=self.opening.open_file(self.status_label,self.line_data,self.canvas_data)
             
    def open_reference(self):
        self.reference,lambdas=self.opening.open_reference(self.status_label,self.line_reference,self.canvas_reference)   
        
    def rescale_reference(self):
        self.reference=self.data_manipulation.rescale_reference(self.intensities,self.lambdas,self.reference,self.status_label,self.line_reference,self.canvas_reference)
#    
    def substrate_subtraction(self):
        self.intensities=self.data_manipulation.subtract_substrate(self.reference,self.intensities,self.progressbar,self.status_label,self.lambdas,self.line_data,self.canvas_data)
 
    def clean_data(self):
        self.lambdas,self.intensities=self.data_manipulation.clean_data(self.intensities,self.master,self.progressbar,self.status_label,self.lambdas,self.line_data,self.canvas_data)
        
    def fit2D(self):
        self.map_FWHM2D,self.map_position_2D=self.fitting.fit_perform('2D',self.intensities,self.lambdas,
                                              self.progressbar,self.status_label,
                                              self.position_x,self.position_y)
        
    def fitG(self):
        self.map_FWHMG,self.map_position_G=self.fitting.fit_perform('G',self.intensities,self.lambdas,
                                              self.progressbar,self.status_label,
                                              self.position_x,self.position_y)
    def fitD(self):
        self.map_FWHMG,self.map_position_G=self.fitting.fit_perform('D',self.intensities,self.lambdas,
                                              self.progressbar,self.status_label,
                                              self.position_x,self.position_y)
  
    def fitSi_H(self):
        self.map_FWHMG,self.map_position_G=self.fitting.fit_perform('Si-H',self.intensities,self.lambdas,
                                              self.progressbar,self.status_label,
                                              self.position_x,self.position_y)


class Opening:
    
    def __init__(self):
        self.position_x=[]
        self.position_y=[]
    
    def open_file(self,status_label,line_data,canvas_data):
        status_label.set("Opening file")
        
        #finction1- open file
        name = askopenfilename(initialdir="C:/Users/",
                           filetypes =(("Text File", "*.txt"),("All Files","*.*")),
                           title = "Choose a file.")  
        df=pd.read_csv(name,names=['x','y','lambda','int'],sep='\t')
        table=df.iloc[:,3].values
        lambdas,rows,index_num=self.get_lambdas(df)
        position_x=df.iloc[:,0].unique()
        position_y=df.iloc[:,1].unique()
          
        intensities=np.reshape(table,(rows,index_num))
        intensities=np.transpose(intensities)
        
        self.verification(df,intensities,index_num,status_label,position_x,position_y)
        intensities=intensities
        lambdas=lambdas[:intensities.shape[0]]
        
        drawing=Drawing()
        drawing.refresh_draw(lambdas,intensities[:,0],line_data,canvas_data)
        return intensities,lambdas,position_x,position_y,rows
    
    def open_reference(self,status_label,line_reference,canvas_reference):
        status_label.set('opening ref')
        name = askopenfilename(initialdir="C:/Users/",
                           filetypes =(("Text File", "*.txt"),("All Files","*.*")),
                           title = "Choose a file."
                           )
        try:
            df_ref=pd.read_csv(name,sep='\t')
        except:
            status_label.set('no file')
        if df_ref.shape[1]==4:
            df_ref=pd.read_csv(name,names=['x','y','lambda','int'],sep='\t')
            table_ref=df_ref.iloc[:,3].values
            lambdas,rows,index_num=self.get_lambdas(df_ref)
            reference=table_ref[:index_num]
            
        elif df_ref.shape[1]==2:
            df_ref=pd.read_csv(name,names=['lambda','int'],sep='\t')
            lambdas=df_ref.iloc[:,0].values
            reference=df_ref.iloc[:,1].values
            
        else:
            status_label.set('wrong format')
        
        reference=reference-reference[0]
        length=np.minimum(reference.shape[0],lambdas.shape[0])
        drawing.refresh_draw(lambdas[:length],reference[:length],line_reference,canvas_reference)
        return reference,lambdas
    
    #function to verify if map is rectangular or not
    def verification(self,df,intensities,index_num,status_label,position_x,position_y):
        x=0
        y=0
        correct=True
        for pos in range(0,intensities.shape[1]):
            if df.iloc[index_num*pos:index_num*pos+1,0].values == x:
                if y != position_y[(pos-1)%position_x.shape[0]]:
                    #print('1 ',x,y,' ',position_y[pos%position_x.shape[0]])
                    correct=False
            elif df.iloc[index_num*pos:index_num*pos+1,1].values == y:
                if x != position_x[(pos-1)%position_y.shape[0]]:
                    #print('2 ',x,y)
                    correct=False
            x=df.iloc[index_num*pos:index_num*pos+1,0].values
            y=df.iloc[index_num*pos:index_num*pos+1,1].values
         
        if correct == True:
            status_label.set('rectangular map')
        else:
            status_label.set('unregular map- You wont make maps of points!')
            
    #function to calculate x-scale (lambda) in spectrum
    def get_lambdas(self,data):
        for index, row in data.iterrows():
            if index==0:
                lambda_num=row['lambda']
            elif row['lambda']==lambda_num and index!=0:
                    index_num=index
                    break
        
        return data.loc[:index_num,'lambda'].values,int(data.shape[0]/index_num),index_num

class Data_manipulation:   
    def __init__(self):
        pass
    def subtract_substrate(self,reference,intensities,progressbar,status_label,lambdas,line_data,canvas_data):
        status_label.set("subtracting substrate")
        if reference.shape[0]!=intensities.shape[0]:
            reference=reference[10:-10]
        intensities_new=np.zeros(intensities.shape)
        memory=0
        sum_1=1e12
        for j in range(0,intensities.shape[1]):
            progressbar["value"] = (j/intensities.shape[1])*100
            progressbar.update()
            status_label.set(str(int(j/intensities.shape[1]*100)) + '%')
            
            for i in range (0,300,1):
                result=intensities[:,0]-0.01*i*reference
                if sum_1>(np.sum(np.abs(result[600:900]-result[600]))):
                    sum_1=np.sum(np.abs(result[600:900]-result[600]))
                    memory=i
            intensities_new[:,j]=intensities[:,j]-0.01*memory*reference
        drawing.refresh_draw(lambdas,intensities_new[:,0],line_data,canvas_data)
        progressbar["value"]=0
        return intensities_new

    def clean_data(self,intensities,master,progressbar,status_label,lambdas,line_data,canvas_data):
        
        intensities=self.spike_remover(intensities,status_label)
        lambdas=lambdas[10:-10]
        return self.noise_remover(lambdas,intensities,progressbar,status_label,line_data,canvas_data)
 
    def spike_remover(self,intensities,status_label):
        status_label.set("spike removing")
        df=pd.DataFrame(intensities)
        
        del intensities
        gc.collect()
        
        df_roll_20=df.rolling(20).mean().shift(-10)
        df_roll_4=df.rolling(4).mean().shift(-2)
        df_clear=np.abs(df-df_roll_4)/np.sqrt(np.abs(df_roll_20))

        df_clear=df_clear.fillna(df_clear.mean())
        df_spike=df_clear>7

        df_anti_spike=~df_spike.astype(int)+2
        df_spike=df_spike.astype(int)
        df_corrected=df_spike*df_roll_20+df_anti_spike*df
        df_corrected=df_corrected.iloc[10:-10,:]
        del df_roll_20,df_roll_4,df_clear,df_spike,df_anti_spike,

        scaler = StandardScaler()
        intensities=scaler.fit_transform(df_corrected.loc[:,:].values)
        return intensities
    
    def noise_remover(self,lambdas,intensities,progressbar,status_label,line_data,canvas_data):
        transformer = FastICA(n_components=10,random_state=0)
        X_transformed = transformer.fit_transform(intensities)            
        intensities_cleaned=np.zeros(intensities.shape)            
        status_label.set("constructing data from ICA vectors")
        
        for x in range(0,intensities.shape[1]):
            progressbar["value"] = (x/intensities.shape[1])*100
            progressbar.update()
            status_label.set(str(int(x/intensities.shape[1]*100)) + '%')
            try:
                popt, pcov = curve_fit(self.grafen_plot, X_transformed, intensities[:,x])
                intensities_cleaned[:,x]=self.grafen_plot(X_transformed, *popt)
            except:
                print(x,' fail to fit')
                intensities_cleaned[:,x]=intensities[:,x]
                
        drawing.refresh_draw(lambdas,intensities_cleaned[:,0],line_data,canvas_data)
        progressbar["value"]=0
        return lambdas,intensities_cleaned
        
    def rescale_reference(self,intensities,lambdas,reference,status_label,line_reference,canvas_reference):
            
        status_label.set("rescale ref to map")
        try:
            rescaler=(intensities[800,0]/reference[800])
        except:
            status_label.set("failed!")
        reference=reference*rescaler
        
        length=np.minimum(reference.shape[0],lambdas.shape[0])
        drawing.refresh_draw(lambdas[:length],reference[:length],line_reference,canvas_reference)
     
            
        return reference
        
    def grafen_plot(self,X,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9):
        y=X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3+X[:,4]*a4+X[:,5]*a5
        y+=X[:,6]*a6+X[:,7]*a7+X[:,8]*a8+X[:,9]*a9
        return y
        
class Fitting:
    
    def __init__(self):
        self.index=0
        self.name=''
        self.background=0
        self.intensity=0
        self.position=0
        self.FWHM=0
        self.boundries=0
        
    def fit_perform(self,name,intensities,lambdas,progressbar,status_label,position_x,position_y):
        self.fit_prepare(name,lambdas,progressbar,status_label)
        drawing=Drawing()
        fit_view_axe,data_line,fit_line,canvas_fit_view=drawing.prepare_preview(self.name,lambdas,intensities[:,0])
        fit_data=[data_line,fit_line,canvas_fit_view,fit_view_axe]
        map_position,map_FWHM=self.peak_fitting(intensities,lambdas,fit_data)
        map_FWHM=map_FWHM[1:].reshape(position_x.shape[0],position_y.shape[0])
        map_position=map_position[1:].reshape(position_x.shape[0],position_y.shape[0])
        
        self.draw_map(map_position,'position'+self.name+'map',position_x,position_y)
        self.draw_map(map_FWHM,'FWHM'+self.name+'map',position_x,position_y)
        progressbar["value"]=0
        return map_FWHM,map_position
    
    def fit_prepare(self,name,lambdas,progressbar,status_label):
        self.progressbar=progressbar
        self.status_label=status_label
        
        self.name=name
        if self.name =='2D':
            self.position=2680
            self.FWHM=30
            self.boundries=100
            
        elif self.name =='G':
            self.position=1580
            self.FWHM=15
            self.boundries=50
            
        elif self.name =='D':
            self.position=1340
            self.FWHM=20
            self.boundries=50
        elif self.name =='Si-H':
            self.position=2140
            self.FWHM=1
            self.boundries=30
        
        self.index=(np.abs(lambdas - self.position)).argmin()
        
    
    def prepare_fit_parameters(self,i,lambdas,intensities):
        x=lambdas[self.index-self.boundries:self.index+self.boundries]
        y=intensities[self.index-self.boundries:self.index+self.boundries]
        if y.size==0:
            try:
                x=lambdas[:self.index+self.boundries]
                y=intensities[:self.index+self.boundries]
            except Exception as e:
                print(e)
                try:
                    x=lambdas[self.index-self.boundries:]
                    y=intensities[self.index-self.boundries:]
                except Exception as e:
                    print(e)
                    x,y=lambdas, intensities
                    
            
        self.background=np.min(y)    
        self.background=np.min(y)
        self.intensity=np.abs((np.max(y)-np.min(y))*(x[-1]-x[0]))
        self.postion=x[np.where(y==np.max(y))]
        self.FWHM=40
        popt_saved=[self.background,self.intensity,
                            self.postion,self.FWHM]
    
        bounds=[[popt_saved[0]-3*np.abs(np.max(y)),0*popt_saved[1],popt_saved[2]-40,popt_saved[3]-20],
                 [popt_saved[0]+3*np.abs(np.max(y)),200*popt_saved[1],popt_saved[2]+40,popt_saved[3]+40]]
        return bounds,popt_saved,x,y
                
    def peak_fitting(self,intensities,lambdas,fit_data):
        [data_line,fit_line,canvas_fit_view,fit_view_axe]=fit_data
        map_position=np.empty(1)
        map_FWHM=np.empty(1)
        for i in range (0,intensities.shape[1]):        
                bounds,popt_saved,x,y= self.prepare_fit_parameters(i,lambdas,intensities[:,i])
                try:
                     popt, pcov = curve_fit(self.fit_lorenz, x , y,p0=popt_saved,bounds=bounds)
                    
                except Exception as exc:
                     popt=popt_saved
                     self.status_label.set(exc)
                self.update_fit(data_line,fit_line,x,y,popt,canvas_fit_view,fit_view_axe)
                self.update_progressbar_and_label(i/intensities.shape[1],popt)
                map_position=np.append(map_position,popt[3])
                map_FWHM=np.append(map_FWHM,popt[2])
        return map_position,map_FWHM
    
    def update_progressbar_and_label(self,progress,popt):
        self.progressbar["value"] = progress*100
        self.progressbar.update()
        par=[]
        for o in popt:
            par.append(int(o))
                
        self.status_label.set(str(int(progress*100)) + '%'+str(par))
       
    def draw_map(self,map_params,tite,x_ticks,y_ticks):
        root= tk.Toplevel()
        root.geometry("400x400")
        fig= plt.Figure(figsize=(4,4),dpi=80)
        canvas= FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(side=tk.BOTTOM)
        canvas.draw()
        fit_view_axe = fig.add_subplot(111)
        fit_view_axe.set_title(tite)
        fit_view_axe.matshow(map_params)
        fig.colorbar(fit_view_axe.matshow(map_params))
        fit_view_axe.set_xticklabels(x_ticks)  
        fit_view_axe.set_yticklabels(y_ticks)
        toolbar= NavigationToolbar2Tk(canvas,root)
        canvas.get_tk_widget().pack(side=tk.TOP, expand=False)
        toolbar.update()    
    def update_fit(self,data_line,fit_line,x,y,popt,canvas,fit_view_axe):
        data_line.set_ydata(y)
        data_line.set_xdata(x)
        fit_line.set_ydata(self.fit_lorenz(x,*popt))
        fit_line.set_xdata(x)
        fit_view_axe.set_ylim(np.min(y),np.max(y))
        fit_view_axe.set_xlim(np.min(x),np.max(x))
        canvas.draw()
        
    def fit_lorenz(self,E,y0,I0,E0,gamma):
        return (y0+I0*((gamma**2)/(((E-E0)**2)+gamma**2))) 
        
class Drawing:
    def __init__(self):
        pass
        
        
    def refresh_draw(self,lambdas,intensities,line,canvas):
        x=lambdas
        y=intensities
        line.set_ydata(y)
        line.set_xdata(x)
        figure = canvas.figure.axes[0]
        figure.set_xlim(x.min(), x.max())
        figure.set_ylim(y.min(), y.max())  
        canvas.draw()
           
    def prepare_preview(self,name,lambdas,intensities):
        root_draw = tk.Toplevel()
        fig2 = plt.Figure()
        canvas2 = FigureCanvasTkAgg(fig2, master=root_draw)
        canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1.0)
        fit_view_axe = fig2.add_subplot(111)
        
        x_draw,y_draw=lambdas,intensities
        x_draw_fit,y_draw_fit=lambdas,intensities
    
        fit_view_axe.set_title(name)
        line2D,=fit_view_axe.plot(x_draw,y_draw)
        line2D_fit,=fit_view_axe.plot(x_draw_fit,y_draw_fit)
        #fit_view_axe.plot(x_draw,y_draw)
        canvas2.draw()
        return fit_view_axe,line2D,line2D_fit,canvas2
        
    def draw_master(self,data,text,master):
        fig = Figure(figsize=(5,2), dpi=100)
        ploting = fig.add_subplot(111)
        line,=ploting.plot(data)
        ploting.set_title(text)
        canvas = FigureCanvasTkAgg(fig, master)
        canvas.draw()
        if text=='Data':
            canvas.get_tk_widget().pack(side=tk.TOP, expand=False)
            toolbar = NavigationToolbar2Tk(canvas, master)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, expand=False)
            
        elif text=='Reference':
            canvas.get_tk_widget().pack(side=tk.BOTTOM, expand=False)
            toolbar = NavigationToolbar2Tk(canvas, master)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, expand=False)
        return line,canvas
          





      
root = tk.Tk()
opening=Opening()
data_manipulation=Data_manipulation()
fitting=Fitting()
drawing=Drawing()

root.geometry("700x600")
app = Window(root,opening,data_manipulation,fitting,drawing)
root.mainloop()  