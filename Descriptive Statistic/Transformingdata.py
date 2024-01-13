import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

file_path = 'polity5_cleaned.csv'
df = pd.read_csv(file_path)

def ExponentialandQuadraticplot(x):
    #dependent value
    y = df['durable']

    #Exponential model
    plt.subplot(221)
    plt.title('Exponential transformation')
    plt.scatter(x,y,c="blue" )
    plt.yscale('log')
    plt.ylabel(f"Dependent: {y.name}" )
    plt.xlabel(f"Independent: {x.name}")
    plt.subplot(222)
    plt.title('Residual plot')
    sns.residplot(x=x.name, y=y.name, data=df) 

    #Quadratic model
    plt.subplot(223)
    plt.title('Quadratic transformation')
    plt.scatter(x,np.sqrt(y),c="blue" )

    plt.ylabel(f"Dependent: {y.name}" )
    plt.xlabel(f"Independent: {x.name}")
    plt.subplot(224)
    plt.title('Residual plot')
    sns.residplot(x=x.name, y=y.name, data=df) 

    # To show the plot
    plt.show()

def ReciprocalandLogPlot(x):
    #dependent value
    y = df['durable']
    #reciprocal model
    plt.subplot(221)
    plt.title('Reciprocal transformation')
    plt.scatter(x,1/y,c="blue" )

    plt.ylabel(f"Dependent: {y.name}" )
    plt.xlabel(f"Independent: {x.name}")
    plt.subplot(222)
    plt.title('Residual plot')
    sns.residplot(x=x.name, y=y.name, data=df) 


    #Logarithmic model
    plt.subplot(223)
    plt.title('Logarithmic transformation')
    plt.xscale('log')
    plt.scatter(x,y,c="blue" )

    plt.ylabel(f"Dependent: {y.name}" )
    plt.xlabel(f"Independent: {x.name}")
    plt.subplot(224)
    plt.title('Residual plot')
    sns.residplot(x=x.name, y=y.name, data=df) 
    
    # To show the plot
    plt.show()


def PowerPlot(x):
    #dependent value
    y = df['durable']
    #reciprocal model
    plt.subplot(221)
    plt.title('Power transformation')
    plt.yscale('log')
    plt.xscale('log')
    plt.scatter(x,y,c="blue" )

    plt.ylabel(f"Dependent: {y.name}" )
    plt.xlabel(f"Independent: {x.name}")
    plt.subplot(222)
    plt.title('Residual plot')
    sns.residplot(x=x.name, y=y.name, data=df) 
    
    plt.show()






ExponentialandQuadraticplot(df['xrreg'])
ReciprocalandLogPlot(df['xrreg'])
PowerPlot(df['xrreg'])