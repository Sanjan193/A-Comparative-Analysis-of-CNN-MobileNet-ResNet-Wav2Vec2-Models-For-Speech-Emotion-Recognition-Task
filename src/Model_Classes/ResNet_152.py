# making model class
class Speech_emotion(nn.Module):

  def __init__(self):

     super(Speech_emotion,self).__init__()

    # --------------------------------------------------------------------------------------
    # pretrained block
     self.backbone_model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
     self.backbone_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
     self.backbone_model.fc = nn.Identity() #disabling the classifier
    #  --------------------------------------------------------------------------------------
     self.dense_1 = nn.Linear(2048*1*1,524)
     self.activation1 = nn.ReLU()
     self.batchnorm1 = nn.BatchNorm1d(524)
     self.dropout_1 = nn.Dropout(0.5)

     self.dense_2 = nn.Linear(524,256)
     self.activation2 = nn.ReLU()
     self.batchnorm2 = nn.BatchNorm1d(256)
     self.dropout_2 = nn.Dropout(0.3)

     self.dense_3 = nn.Linear(256,8)
     # using crossentropy as loss function so soft max is not needed here

  def forward(self,x):


    x = self.backbone_model(x)
    x = self.dense_1(x)
    x =  self.batchnorm1(x)
    x = self.activation1(x)
    x = self.dropout_1(x)

    x =  self.dense_2(x)
    x = self.batchnorm2(x)
    x = self.activation2(x)
    x= self.dropout_2(x)

    x =  self.dense_3(x)

    return x

  # -------------------------------------------------------------------------------
  # pretrained
  def CNN_unfreezing(self):
      for param in self.backbone_model.parameters():
        param.requires_grad = True
      print("Loaded model's gradients trancking is enabled for Fine Tuing")

  def CNN_freezing(self):
    for param in self.backbone_model.parameters():
        param.requires_grad = False
    print("Loaded model's gradients trancking is disabled for Training")

    #-----------------------------------------------------------------------
