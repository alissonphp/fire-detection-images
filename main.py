import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import Recall,AUC
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import Xception

sns.set_style('darkgrid')

df = pd.DataFrame(columns=['caminho','rotulo'])

for dirname, _, filenames in os.walk('./fire_dataset/fire_images'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        df = pd.concat([df, pd.DataFrame([[os.path.join(dirname, filename),'fire']],columns=['caminho','rotulo'])])

for dirname, _, filenames in os.walk('./fire_dataset/non_fire_images'):
    for filename in filenames:
        df = pd.concat([df, pd.DataFrame([[os.path.join(dirname, filename),'non_fire']],columns=['caminho','rotulo'])])
        print(os.path.join(dirname, filename))

df = df.sample(frac=1).reset_index(drop=True)

# plotando gráfico com a distribuição das imagens do dataframe
# fig = px.scatter(data_frame = df,x=df.index,y='rotulo',color='rotulo',title='Distribuição das imagens carregadas no dataframe que apresentam fogo e as que não apresentam')
# fig.update_traces(marker_size=2)
# fig.show()

def shaper(row):
    shape = image.load_img(row['caminho']).size
    row['altura'] = shape[1]
    row['largura'] = shape[0]
    return row
df = df.apply(shaper,axis=1)
#print(df.head(10))

# plotando gráfico de distribuição de formas das imagens do dataset que estão no data frame montado
# sns.set_style('darkgrid')
# fig,(ax1,ax2,ax3) = plt.subplots(1,3,gridspec_kw={'width_ratios': [3,0.5,0.5]},figsize=(8,6))
# sns.kdeplot(data=df.drop(columns=['caminho','rotulo']),ax=ax1,legend=True)
# sns.boxplot(data=df,y='altura',ax=ax2,color='skyblue')
# sns.boxplot(data=df,y='largura',ax=ax3,color='orange')
# plt.suptitle('Distribuição das imagens a partir das formas (dimensões e densidade)')
# ax3.set_ylim(0,7000)
# ax2.set_ylim(0,7000)
# plt.tight_layout()
# plt.show()

generator = ImageDataGenerator(
    rotation_range= 20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range = 2,
    zoom_range=0.2,
    rescale = 1/255,
    validation_split=0.2,
)

train_gen = generator.flow_from_dataframe(df,x_col='caminho',y_col='rotulo',images_size=(256,256),class_mode='binary',subset='training')
val_gen = generator.flow_from_dataframe(df,x_col='caminho',y_col='rotulo',images_size=(256,256),class_mode='binary',subset='validation')

# definindo o dicionário de classificação das imagens

class_indices = {}
for key in train_gen.class_indices.keys():
    class_indices[train_gen.class_indices[key]] = key
    
# print(class_indices)
# definição do modelo e pilha de camadas aplicadas
model = Sequential()
model.add(Conv2D(filters=32,kernel_size = (2,2),activation='relu',input_shape = (256,256,3)))
model.add(MaxPool2D())
model.add(Conv2D(filters=64,kernel_size=(2,2),activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(filters=128,kernel_size=(2,2),activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))
model.summary()

# compilando o modelo utilizando o otimizador adam
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy',Recall(),AUC()])

# definindo as chamadas de retorno para controle dos parâmetros de perda com objetivo de garantir uma alta acurácia e baixo de índice de loss
early_stoppping = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5)

model.fit(x=train_gen,batch_size=32,epochs=15,validation_data=val_gen,callbacks=[early_stoppping,reduce_lr_on_plateau])

# avaliando o modelo atual
# eval_list = model.evaluate(val_gen,return_dict=True)
# for metric in eval_list.keys():
#     print(metric+f": {eval_list[metric]:.2f}")

# testando o modelo de predição para identificação de foco de incêndio na amazônia a partir de imagem aérea
amazonia = image.load_img('./examples/incendio.jpg')
# aplicando ajustes e processamento na imagem
img = image.img_to_array(amazonia)/255
img = tf.image.resize(img, (256,256))
img = tf.expand_dims(img, axis=0)

teste_modelo = int(tf.round(model.predict(x=img)).numpy()[0][0])
print("O retorno da análise preditiva do modelo é: ", teste_modelo, " e a imagem foi classificada como: ", class_indices[teste_modelo])