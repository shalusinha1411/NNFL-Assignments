import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def trainer(model, train_dataloader, val_dataloader, num_epochs, path_to_save='/home/atharva',
          checkpoint_path='/home/atharva',
          checkpoint=100, train_batch=1, test_batch=1, device='cuda:0'): # 2 Marks. 
    """
    Everything by default gets shifted to the GPU. Select the device according to your system configuration
    If you do no have a GPU, change the device parameter to "device='cpu'"
    :param model: the Classification model..
    :param train_dataloader: train_dataloader
    :param val_dataloader: val_Dataloader
    :param num_epochs: num_epochs
    :param path_to_save: path to save model
    :param checkpoint_path: checkpointing path
    :param checkpoint: when to checkpoint
    :param train_batch: 1
    :param test_batch: 1
    :param device: Defaults on GPU, pass 'cpu' as parameter to run on CPU. 
    :return: None
    """
    #torch.backends.cudnn.benchmark = True #Comment this if you are not using a GPU...
    # set the network to training mode.
    model.cuda()  # if gpu available otherwise comment this line. 
    # your code goes here. 
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion=nn.CrossEntropyLoss().cuda()
    max_accuracy=None
    
    training_loss=[]
    val_loss=[]
    training_acc=[]
    val_acc=[]
    for epoch in range(num_epochs):
        
        if ((epoch+1)%checkpoint)==0:
            torch.save({
                'epoch':epoch,
                'optimizer': optimizer.state_dict(),
                'train_loss': training_loss,
                'val_loss': val_loss,
                'train_acc' : training_acc,
                'val_acc' : val_acc
            }, checkpoint_path+'/checkpoint.pt')
            exit(10)
            
        epoch_train_loss = 0
        epoch_accuracy_train = 0

        for _, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            statement = data['statement'].to(device)
            justification = data['justification'].to(device)
            credit_history = data['credit_history'].to(device)
            label = data['label'].to(device)
            #torch.tensor.to("device")
            output=model(statement,justification,credit_history)
            #__, predicted = torch.max(output.data, 1)
            loss=criterion(output,label)
            
            epoch_acc_train += (output == label).sum().item()
            loss.backward()
            optimizer.step()
            epoch_train_loss+=loss.item()
            epoch_accuracy_train+= (output == label).sum().item()
            del statement,justification,credit_history,label
        training_loss.append(epoch_train_loss / ((_+1)*train_batch)) 
        training_acc.append(epoch_acc_train/((_+1)*train_batch)) 

        with torch.no_grad():
            model.eval()
            epoch_val_loss=0
            epoch_val_accuracy=0


            for _, data in enumerate(val_dataloader):
                statement = data['statement'].to(device)
                justification = data['justification'].to(device)
                credit_history = data['credit_history'].to(device)
                label = data['label'].to(device)
                #torch.tensor.to("device")
                output_val=model(statement,justification,credit_history)
                loss=criterion(output_val,label)
                epoch_val_loss+=loss.item()
                epoch_val_accuracy += (output_val == label).sum().item()
            
            
    
            val_loss.append(epoch_val_loss / ((_+1)*test_batch)) 
            val_acc.append(epoch_val_accuracy/((_+1)*test_batch))
            if(max_accuracy== None):
              max_accuracy=epoch_val_accuracy/((_+1)*test_batch)
              
            elif(epoch_val_accuracy/_*test_batch) > max_accuracy:
                    torch.save(model.state_dict(), path_to_save +'/model.pth')
                
    

    plt.plot(training_acc)
    plt.plot(val_acc)
    plt.plot(training_loss)
    plt.plot(val_loss)
    plt.show()