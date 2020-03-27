# NBEATS

##### N-BEATS: Neural basis expansion analysis for interpretable time series forecasting

NBEATS is a pytorch based library for deep learning based time series forecasting (https://arxiv.org/pdf/1905.10437v3.pdf) and utilises nbeats-pytorch.

### Dependencies: Python >=3.6

### Installation

```sh
$ pip install NBEATS
```

#### Import
```sh
from NBEATS import NeuralBeats
```



Mandatory Parameters:
- data
- forecast_length 

Basic model with only mandatory parameters can be used to get forecasted values as shown below:
```sh
import pandas as pd
from NBEATS import NeuralBeats

data = pd.read_csv('test.csv')   
data = data.values        # (nx1 array)

model = NeuralBeats(data=data, forecast_length=5)
model.fit()
forecast = model.predict()
```


#### Optional parameters to the model object
| Parameter | Default Value|
| ------ | --------------|
| backcast_length | 3* forecast_length |
| path | '  ' (path to save intermediate training checkpoint) |
| checkpoint_name | 'NBEATS-checkpoint.th'| 
| mode| 'cpu'| Any of the torch.device modes|
| batch_size | len(data)/10 |
| thetas_dims | [4, 8] | 
| nb_blocks_per_stack | 3 |
| share_weights_in_stack | False |
| train_percent |  0.8 |
| save_model | False |
| hidden_layer_units | 128 |
| stack | [1,1] (As per the paper- Mapping is as follows -- 1: GENERIC_BLOCK,  2: TREND_BLOCK , 3: SEASONALITY_BLOCK)|


#### Functions

#### fit() 

This is used for training the model. The default value of parameters passed are epoch=25, optimiser=Adam, plot=False, verbose=True


ex:

```sh

model.fit(epoch=25,optimiser=torch.optim.AdamW(model.parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.01, amsgrad=False),plot=True, verbose=True)

```


###### predict_data ()

The argument to the method could be empty or a numpy array of length backcast_length x 1 which means if no argument is passed and training data is till Dec 2019 the prediction will begin from Jan 2020 but if forcasting for 3 months ahead(forecast_length=3)from March 2020 then numpy array of backcast_length (3 x forecast_length -This is by default) i.e 9(3 x 3) previous months (June 2019 to Feb 2020) needs to be provided to predict for March,Apr,May 2020.

Important Note : Backcast length can be provided as a model argument along with forecast_length  eg backcast_length=6,backcast_length=9,backcast_length=12......till backcast_length=21 for forecast_length=3 ,as the paper suggests values between 2 x forecast_length  to 7 x forecast_length .The default is 3 x forecast_length .
 
Returns forecasted values.

#### save(file) & load(file,optimizer):
Save and load the model after training respectively. 

Example: model.save('NBEATS.th') or model.load('NBEATS.th')



## DEMO

 1: GENERIC_BLOCK and 3: SEASONALITY_BLOCK stacks are used below (stack=[1,3]).Go through th paper for more details.Playing around with the 3 blocks(GENERIC,SEASONALITY and TREND) might improve accuracy.
```sh
import pandas as pd
from NBEATS import NeuralBeats
from torch import optim

data = pd.read_csv('test.csv')   
data = data.values # nx1(numpy array)

model=NeuralBeats(data=data,forecast_length=5,stack=[1,1],nb_blocks_per_stack=3,thetas_dims=[3,7])

#or use prebuilt models
#model.load(file='NBEATS.th')


#use customised optimiser with parameters
model.fit(epoch=35,optimiser=optim.AdamW(model.parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.01, amsgrad=False)) 
#or 
#model.fit()

forecast=model.predict()
#or
#model.predict(predict_data=pred_data) where pred_data is numpy array of size backcast_length*1
```


