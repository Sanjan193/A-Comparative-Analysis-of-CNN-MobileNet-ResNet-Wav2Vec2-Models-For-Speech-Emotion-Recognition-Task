# model evaluation 

check = torch.load("best_model_path.pth")
final_model.load_state_dict(check)
final_model.CNN_unfreezing()
final_model.eval()

with torch.no_grad():
  count = 0
  avg_accuracy = 0
  avg_recall = 0
  avg_precition = 0
  for data,label in val_set:

      count += 1
      data = data.to(device)

      op = final_model(data)
      output = torch.argmax(op,dim=1)

      model_accuracy = accuracy_score(output.cpu().numpy(),label.numpy())
      model_recall = recall_score(output.cpu().numpy(),label.numpy(),average='weighted')
      model_precition = precision_score(output.cpu().numpy(),label.numpy(),average='weighted')
      avg_accuracy += model_accuracy
      avg_recall += model_recall
      avg_precition += model_precition
      # print(f"model accuracy for batch {count} :",model_accuracy)
  print("Average Accuracy for test set : ",(avg_accuracy/count))
  print("Average Recall for test set : ",(avg_recall/count))
  print("Average Precition for test set : ",(avg_precition/count))

