import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

data = pd.read_csv(os.path.join('../sources/airline-passengers.csv'))
print(data.head())
print(data.info())

# create graph
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Passengers'])
plt.xlabel('t, мин') #Подпись для оси х
plt.ylabel('НПС') #Подпись для оси y
plt.title('График накопленного числа событий') #Название
plt.grid()
plt.show()