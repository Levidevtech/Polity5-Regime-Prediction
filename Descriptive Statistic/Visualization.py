import matplotlib.pyplot as plt
import pandas as pd


file_path = 'polity5_cleaned.csv'
df = pd.read_csv(file_path)

def scatterplot(independentvariable):
    #dependent value
    y = df['durable']
    if isinstance(independentvariable, list):
        subplot = 221
        for item in independentvariable:
            x = item
            plt.subplot(subplot)
            #scatterplot with blue dots
            plt.scatter(x,y,c="blue")
            plt.ylabel(f"Dependent: {y.name}" )
            plt.xlabel(f"Independent: {x.name}")
            subplot+=1 
    else:    
        #independent value
        x = independentvariable
        #scatterplot with blue dots
        plt.scatter(x,y,c="blue")
        plt.ylabel(f"Dependent: {y.name}" )
        plt.xlabel(f"Independent: {x.name}")

    # To show the plot
    plt.show()


listofvariables = [df['xrreg'], df['xrcomp'], df['xropen'], df['xconst']]
secondlist = [df['parreg'], df['parcomp'], df['exrec'], df['exconst']]
last = df['polcomp']
scatterplot(last)

