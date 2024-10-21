import torch
import lightning as L
from tfc import NTXentLoss_poly, TFC_Model, TFC_Backbone
from dataloader import ReyesModule

def train():
    # Build the pretext model, the pretext datamodule, and the trainer
    pretext_model = TFC_Model(backbone=TFC_Backbone(), pred_head=None, loss = NTXentLoss_poly("cuda", 8, 0.2, True)) # batch size 8
    pretext_datamodule = ReyesModule(root_data_dir=f"../dataset/UCI/", batch_size=8)
    lightning_trainer = L.Trainer(
        accelerator="gpu",
        # max_epochs=40,
        max_epochs=1,
        max_steps=-1,
        enable_checkpointing=True, 
        logger=True)

    # Fit the pretext model using the pretext_datamodule
    lightning_trainer.fit(pretext_model, pretext_datamodule)

    # Save the backbone weights
    torch.save(pretext_model.backbone.state_dict(), "weights/pretrained_backbone_weights.pth")

if __name__ == "__main__":
    train()




