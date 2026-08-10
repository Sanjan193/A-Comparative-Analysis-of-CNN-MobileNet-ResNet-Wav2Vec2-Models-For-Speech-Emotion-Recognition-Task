# making the feeding a function
def model_feed(epochs,
               train_total_loss_per_batch,
               train_loss_array_per_batch,
               val_total_loss_per_batch,
               val_loss_array_per_batch,
               train_accuracy,
               val_accuracy,
               loader_1,
               model,
               optimizer,
               loss,
               f1,
               device,
               loader_2,
               accuracy_score,
               f1_score,
               scheduler,
               early_stop):
   best_val_accuracy = 0
   for epoch in range(epochs):

        train_total_loss_per_batch = 0
        avg_train_loss =0
        f1_train  = 0
        f1_train_avg = 0
        accuracy_train = 0
        accuracy_train_avg = 0

        val_total_loss_per_batch = 0
        avg_val_loss = 0
        model_val_f1 =0
        model_val_f1_avg =0
        model_val_accuracy =0
        model_val_accuracy_avg = 0

        # training part
        model.train()
        for speech,label_1 in loader_1:
             optimizer.zero_grad()

             speech,label_1 = speech.to(device),label_1.to(device)

             train_output = model(speech)
             losses = loss(train_output,label_1)
             train_total_loss_per_batch = train_total_loss_per_batch + losses.item()
             f1_train = f1_train + f1_score(label_1.cpu().detach().numpy(),
                                            (torch.argmax(train_output.cpu().detach(),dim=1)).numpy(),average='weighted')
             accuracy_train = accuracy_train + accuracy_score(label_1.cpu().detach().numpy(),
                                                              (torch.argmax(train_output.cpu().detach(),dim=1)).numpy())

             losses.backward()
             optimizer.step()
        # -----------------------------------------------------------------
        # validation part
        model.eval()
        with torch.no_grad():
         for data,label_2 in loader_2:

            data,label_2 = data.to(device),label_2.to(device)

            op = model(data)
            loss_val = loss(op,label_2)
            val_total_loss_per_batch = val_total_loss_per_batch +loss_val.item()
            val_op = torch.argmax(op,dim=1)
            model_val_accuracy = accuracy_score(label_2.cpu().numpy(),(val_op.cpu().detach()).numpy()) + model_val_accuracy
            model_val_f1 =model_val_f1 + f1_score(label_2.cpu().numpy(),
                                                  (val_op.cpu().detach()).numpy(),average='weighted')



        avg_train_loss = train_total_loss_per_batch/len(loader_1)
        f1_train_avg = f1_train/len(loader_1)
        accuracy_train_avg = (accuracy_train/len(loader_1))

        model_val_accuracy_avg  = model_val_accuracy /len(loader_2)
        model_val_f1_avg = model_val_f1/len(loader_2)
        avg_val_loss = val_total_loss_per_batch/len(loader_2)


        train_accuracy.append(accuracy_train_avg)
        val_accuracy.append(model_val_accuracy_avg)
        # enabling the scheduler
        scheduler.step(model_val_accuracy_avg)

        if epoch % 1 == 0 :
                print(f"\n{'='*85}")
                print(f"Epoch {epoch}")
                print(f"{'Training':<40} {'Validation':<40}")
                print(f"{'-'*85}")

                print(f"{'Loss':<15}: {avg_train_loss}    "
                      f"{'Loss':<15}: {avg_val_loss}")

                print(f"{'Accuracy':<15}: {accuracy_train_avg}    "
                      f"{'Accuracy':<15}: {model_val_accuracy_avg}")

                print(f"{'F1 Score':<15}: {f1_train_avg}    "
                      f"{'F1 Score':<15}: {model_val_f1_avg}")

                print(f"{'Current LR':<15}: {optimizer.param_groups[0]['lr']}   ")

                print(f"{'='*85}")
                train_loss_array_per_batch.append(avg_train_loss)
                val_loss_array_per_batch.append(avg_val_loss)

        # saving the best validation accuracy model
        if  model_val_accuracy_avg>best_val_accuracy:
          best_val_accuracy=  model_val_accuracy_avg
          torch.save(model.state_dict(),"best_model_path.pth")


        early_stop.early_stopping_check(avg_val_loss)

        if early_stop.stop:
          print("Stopping Training")
          break


