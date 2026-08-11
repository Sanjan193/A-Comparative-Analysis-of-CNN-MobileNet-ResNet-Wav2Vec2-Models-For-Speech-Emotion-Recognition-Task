# splitting the data
x_train,x_test,y_train,y_test = train_test_split(X_all_data,Y_all_label,test_size=400)

x_final_test,x_valid,y_final_test,y_valid = train_test_split(x_test,y_test,test_size =200)



# making these tensors because I used GPU
dataset_train = TensorDataset(x_train,y_train)
dataset_test = TensorDataset(x_final_test,y_final_test)
dataset_valid = TensorDataset(x_valid,y_valid)

# defining batch
torch.manual_seed(42)
train_set = DataLoader(dataset_train, batch_size=34,shuffle=True)
test_set = DataLoader(dataset_test, batch_size=54,shuffle=False)
val_set = DataLoader(dataset_valid, batch_size=54,shuffle=False)
