## main command

### train
python train.py

### Test
python test.py --weights weights/last.pth

### Demo 

python demo.py --source inference/images --weights weights/last.pth

----------------------------------------------------------------

## dataset 
在/data中的full.yaml中更改圖片及GT路徑及類別，
也可以新增其他yaml檔案<br>
在 train.py test.py demo.py中更改你的yaml路徑
```python
parser.add_argument('--data', type=str, default='data/full.yaml', 
                                            help='dataset yaml path')
```
----------------------------------------------------------------
## environment
```
conda create --name myenv python=3.9
conda activate myenv

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
會報torch的錯不用理他

pip install opencv-python
pip install pyyaml
pip install tensorboardX
pip install tqdm
pip install prefetch_generator
pip install matplotlib
pip install timm
pip install fvcore
pip install thop
pip install pandas
```

---------------
## TOOL
### Tensorboard
tensorboard --logdir=runs


---------------

## 新增model

這邊以<a href="https://github.com/milesial/Pytorch-UNet">老師上課使用的Unet</a>舉例

1. 請自行clone下來使用不要改我桌面資料夾的內容
```
git clone https://github.com/richardkuo1999/2023_ITRI_future_eagle_Group_5.git
```
2. 在/models中新增一個Unet.py
3. 將<a href="https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py">Unet</a>的網路架構全部複製過來
4. 注意一下model通常會包成一塊一塊放在其他.py檔內，將有用到的全部複製到Unet.py中

```python
""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
```

5. 確認model的輸入如Unet是 n_channels, n_classes, bilinear

```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
```

6. 在/models/Unet.py最下面新增，並且**self.model = UNet(ch, nc)**的輸入與步驟四的輸入對上。</br>
(bilinear已在 class Unet預設為False如果想用雙線性插值進行上取樣可改為self.model = UNet(ch, nc, True))

```python
def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
        # elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

class Model(nn.Module):
  def __init__(self, cfg = None, nc = 2, ch=3, ):
    super(Model, self).__init__()
    self.model = UNet(ch, nc)

    # Init weights, biases
    initialize_weights(self)
  def forward(self, x):
        return self.model(x)
  ```

7. 到models/model.py中的 build_model新增你的model

```python
if cfg == 'UNext':
  from models.UNext import Model
elif cfg == 'Unet':
  from models.UNext import Model
elif cfg == 'Newmodel':
  from models.Newmodel import Model
else:
    raise Exception(f'model {cfg} not exist')
```

8. 根據你要做的事在train.py test.py demo.py更改cfg參數為你要的model名稱或yaml路徑

```python
parser.add_argument('--cfg', type=str, default='Unet', 
                                      help='model yaml path')
```

9. 開始train

```
請記得用nvidia-smi檢查有沒有人在使用顯卡
若有請用，選擇其他張享卡訓練
python train.py --device 0~2
```

10. 確定可以用後歡迎PR到我的專案中
