# making model class
class Speech_emotion(nn.Module):

  def __init__(self):

     super(Speech_emotion,self).__init__()

    # --------------------------------------------------------------------------------------
    # the CNN Block
     self.conv = nn.Sequential(

        # CONVO Lyaer 1
        nn.Conv2d(1,70,kernel_size=8,padding=1,stride=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=1),


        # CONVO Lyaer 2
        nn.Conv2d(70,192,kernel_size=7,padding=1,stride=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=1),


        # CONVO Lyaer 3
        nn.Conv2d(192,250,kernel_size=4,padding=1,stride=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=1)


    )
     self.average_pooling = nn.AdaptiveAvgPool2d(4)

     self.dense_1 = nn.Linear(250*4*4,524)
     self.activation1 = nn.ReLU()
     self.batchnorm1 = nn.BatchNorm1d(524)
     self.dropout_1 = nn.Dropout(0.5)

     self.dense_2 = nn.Linear(524,256)
     self.activation2 = nn.ReLU()
     self.batchnorm2 = nn.BatchNorm1d(256)
     self.dropout_2 = nn.Dropout(0)

     self.dense_3 = nn.Linear(256,8)
     # using crossentropy as loss function so soft max is not needed here

  def forward(self,x):


    x=self.conv(x)
    x=self.average_pooling(x)
    x= torch.flatten(x,1)

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


  def CNN_unfreezing(self):
      for param in self.conv.parameters():
        param.requires_grad = True
      print("Loaded model's gradients trancking is enabled for Fine Tuing")

  def CNN_freezing(self):
    for param in self.conv.parameters():
        param.requires_grad = False
    print("Loaded model's gradients trancking is disabled for Training")
