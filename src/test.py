import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import random

plt.switch_backend('agg')

f=open("./id_to_data","rb+")
data=pickle.load(f)

f=open("./id_to_box","rb+")
box=pickle.load(f)

f=open("./id_to_size","rb+") 
size=pickle.load(f)

index=[i for i in range(11788)]
index=random.sample(index,100)


model=keras.models.load_model("./model.h5")
result=model.predict(data[index,:,:,:])

mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]
j=0
for i in index:
    print("Predicting "+str(i)+"th image.")
    true_box=box[i]
    image=data[i]
    prediction=result[j]
    j+=1
    for channel in range(3):
        image[:,:,channel]=image[:,:,channel]*std[channel]+mean[channel]

    image=image*255
    image=image.astype(np.uint8)
    plt.imshow(image)


    plt.gca().add_patch(plt.Rectangle((true_box[0],true_box[1]),true_box[2],true_box[3],fill=False,edgecolor='red',linewidth=2,alpha=0.5))
    plt.gca().add_patch(plt.Rectangle((prediction[0]*224,prediction[1]*224),prediction[2]*224,prediction[3]*224,fill=False,edgecolor='green',linewidth=2,alpha=0.5))
    plt.show()
    plt.savefig("./prediction/"+str(i)+".png")
    plt.cla()



