# making model class
class Speech_emotion(nn.Module):

  def __init__(self):

     super(Speech_emotion,self).__init__()

    # --------------------------------------------------------------------------------------
    # pretrained block
     self.backbone_model = Wav2Vec2Model.from_pretrained("superb/wav2vec2-base-superb-er")

     self.dense_1 = nn.Linear(self.backbone_model.config.hidden_size,524)
    #  --------------------------------------------------------------------------------------
    #  The Classifier
     self.activation1 = nn.ReLU()
     self.batchnorm1 = nn.BatchNorm1d(524)
     self.dropout_1 = nn.Dropout(0.3)

     self.dense_2 = nn.Linear(524,256)
     self.activation2 = nn.ReLU()
     self.batchnorm2 = nn.BatchNorm1d(256)
     self.dropout_2 = nn.Dropout(0.1)

     self.dense_3 = nn.Linear(256,8)
     # using crossentropy as loss function so soft max is not needed here

  def forward(self,x):


    # -------------------------------------------------------------------------------------
    # the model returns here a shape of (batch size,time steps,features(768))
    # so for each audio there will be (time steps,features(768)) dimentioned map

    '''to make this linear I'm taking the mean and max of each waves
     per 'time steps' and then concating the min and max array dise by side to get 1536'''
    x = self.backbone_model(x)
    hidden_state = x.last_hidden_state
    mean_values = torch.mean(hidden_state,dim=1)
    x = self.dense_1(mean_values)
    # --------------------------------------------------------------------------------------
    # Classifier
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
  # Loaded model freezing and unfreezing
  def CNN_unfreezing(self):
      for param in self.backbone_model.parameters():
        param.requires_grad = True
      print("Loaded model's gradients trancking is enabled for Fine Tuing")

  def CNN_freezing(self):
    for param in self.backbone_model.parameters():
        param.requires_grad = False
    print("Loaded model's gradients trancking is disabled for Training")
