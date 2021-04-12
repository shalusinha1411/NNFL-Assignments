# 0.75 Marks. 
# To test your trainer and  arePantsonFire class, Just create random tensor and see if everything is working or not.  
from torch.utils.data import DataLoader

# Your code goes here.
from datasets import  dataset
import Encoder
import Attention
from LiarLiar import arePantsonFire
import trainer
from utils import infer
liar_dataset_train=dataset(prep_Data_from='train')
liar_dataset_val=dataset(prep_Data_from='val')
sentence_length,justification_length = liar_dataset_train.get_max_lenghts()

dataloader_train=DataLoader(liar_dataset_train,batch_size=50)
dataloader_val=DataLoader(liar_dataset_val,batch_size=25)

statement_encoder=Encoder.Encoder(conv_layers=5,hidden_dim=512)
justification_encoder=Encoder.Encoder(conv_layers=5,hidden_dim=512)
multiheadAttention= Attention.MultiHeadAttention(hid_dim=512,n_heads=32)
positionFeedForward=Attention.PositionFeedforward(hid_dim=512,feedForward_dim=2048)
model=arePantsonFire(statement_encoder, justification_encoder,multiheadAttention,positionFeedForward,512,sentence_length,justification_length,200,device='cuda:0')

trainer.trainer(model,dataloader_train,dataloader_val,1,train_batch=1,test_batch=1)

# Do not change module_list , otherwise no marks will be awarded
module_list = [liar_dataset_train, liar_dataset_val, dataloader_train, dataloader_val, statement_encoder, justification_encoder, multiheadAttention, positionFeedForward, model]
del  liar_dataset_val, liar_dataset_train, dataloader_train, dataloader_val


liar_dataset_test = dataset(prep_Data_from='test')
test_dataloader = DataLoader(dataset=liar_dataset_test, batch_size=1)
infer(model=model, dataloader=test_dataloader)
