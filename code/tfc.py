import torch
from torch import nn
import lightning as pl
from typing import List, Tuple
import numpy as np
import torch.fft as fft
from dataloader import ReyesModule
import lightning as L


class TFC_Model(pl.LightningModule):
    def __init__(self, backbone = None, pred_head = True, loss = None, learning_rate = 3e-4, transform = None):
        super(TFC_Model, self).__init__()
        if backbone:
            self.backbone = backbone
        else:
            self.backbone = TFC_Backbone()
        if pred_head:
            self.pred_head = TFC_PredicionHead(num_classes=7)
        else:
            self.pred_head = None

        if loss:
            self.loss_fn = loss
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            
        self.learning_rate = learning_rate
        
        if transform:
            self.transform = transform
        else:
            self.transform = TFC_Transforms()
    
    def forward(self, x_t, x_f = None, all=False):
        if x_f is None:
            x_f = fft.fft(x_t).abs()
        h_t, z_t, h_f, z_f = self.backbone(x_t, x_f)
        if self.pred_head:
            fea_concat = torch.cat((z_t, z_f), dim=1)
            pred = self.pred_head(fea_concat)
            if all:
                return pred, h_t, z_t, h_f, z_f
            return pred
        else:
            return h_t, z_t, h_f, z_f

    def training_step(self, batch, batch_index):
        x = batch[0]
        # verifica onde esta x, cuda ou gpu
        x = x.to(self.device)
        
        labels = batch[1]
        data, aug1, data_f, aug1_f = self.transform(x) 

        if self.pred_head:
            pred = self.forward(data,data_f)
            labels = labels.long()
            loss = self.loss_fn(pred, labels)
        else:
            h_t, z_t, h_f, z_f = self.forward(data, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1, aug1_f)
            loss_t = self.loss_fn(h_t, h_t_aug)
            loss_f = self.loss_fn(h_f, h_f_aug)
            l_TF = self.loss_fn(z_t, z_f)
            l_1, l_2, l_3 = self.loss_fn(z_t, z_f_aug), self.loss_fn(z_t_aug, z_f), self.loss_fn(z_t_aug, z_f_aug)
            loss_c = (1+ l_TF -l_1) + (1+ l_TF -l_2) + (1+ l_TF -l_3)
            lam = 0.2
            loss = lam *(loss_t + loss_f) + (1- lam)*loss_c
        self.log("train_loss", loss, prog_bar=False)
        return loss
        
    def validation_step(self, batch, batch_index):
        # calcula a entropia cruzada
        x = batch[0]
        labels = batch[1]
        data, aug1, data_f, aug1_f = self.transform(x)
        if self.pred_head:
            pred = self.forward(data,data_f)
            labels = labels.long()
            loss = self.loss_fn(pred, labels)
        else:
            h_t, z_t, h_f, z_f = self.forward(data, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1, aug1_f)
            loss_t = self.loss_fn(h_t, h_t_aug)
            loss_f = self.loss_fn(h_f, h_f_aug)
            l_TF = self.loss_fn(z_t, z_f)
            l_1, l_2, l_3 = self.loss_fn(z_t, z_f_aug), self.loss_fn(z_t_aug, z_f), self.loss_fn(z_t_aug, z_f_aug)
            loss_c = (1+ l_TF -l_1) + (1+ l_TF -l_2) + (1+ l_TF -l_3)
            lam = 0.2
            loss = lam *(loss_t + loss_f) + (1- lam)*loss_c
        self.log("val_loss", loss, prog_bar=True)
        return loss
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.99), weight_decay=3e-4)

class TFC_Backbone(nn.Module):
    def _calculate_fc_input_features(self, backbone: torch.nn.Module, input_shape: Tuple[int, int, int]) -> int:
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)
    
    def __init__(self, input_channels = 9, TS_length = 128):
        super(TFC_Backbone, self).__init__()
        self.conv_block_t = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, 60, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block_f = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, 60, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_t = nn.Sequential(
            nn.Linear(self._calculate_fc_input_features(self.conv_block_t, (input_channels, TS_length)), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(self._calculate_fc_input_features(self.conv_block_t, (input_channels, TS_length)), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x_in_t, x_in_f):
        x = self.conv_block_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)
        z_time = self.projector_t(h_time)

        f = self.conv_block_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq

class TFC_PredicionHead(nn.Module):
    def __init__(self, num_classes, conexoes=2):
        super(TFC_PredicionHead, self).__init__()
        if conexoes != 2:
            print(f"Apenas um ramo sera utilizado: {conexoes}")
        self.logits = nn.Linear(conexoes*128, 64)
        self.logits_simple = nn.Linear(64, num_classes)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
    

class TFC_Transforms:
    def __init__(self):
        self.report_change_sample= True
        pass

    def __call__(self, x):

        if x.shape[-1] != 128:
            if self.report_change_sample:
                print(f"Sample size is not 128, changing it to 128 by interpolation.")
                self.report_change_sample = False
            # se o tamanho do sinal for diferente de 128, deve ser feito uma reamostragem por interpolação
            x = nn.functional.interpolate(x, size=128, mode='linear')
            
        device = x.device
        tipo = type(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).type(torch.FloatTensor)
        else:
            x = x.type(torch.FloatTensor)
        freq = fft.fft(x).abs()
        y1 = self.DataTransform_TD(x)
        y2 = self.DataTransform_FD(freq)
        x, y1, freq, y2 = x.type(tipo), y1.type(tipo), freq.type(tipo), y2.type(tipo)
        x, y1, freq, y2 = x.to(device), y1.to(device), freq.to(device), y2.to(device)
        return x, y1, freq, y2


    def one_hot_encoding(self, X, n_values=None):
        X = [int(x) for x in X]
        if n_values is None:
            n_values = np.max(X) + 1
        b = np.eye(n_values)[X]
        return b

    def DataTransform_TD(self, sample, jitter_ratio = 0.8):
        """Weak and strong augmentations"""
        aug_1 = self.jitter(sample, jitter_ratio)

        li = np.random.randint(0, 4, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
        li_onehot = self.one_hot_encoding(li)
        aug_1[1-li_onehot[:, 0]] = 0 # the rows are not selected are set as zero.
        return aug_1


    def DataTransform_FD(self, sample):
        """Weak and strong augmentations in Frequency domain """
        aug_1 =  self.remove_frequency(sample, 0.1)
        aug_2 = self.add_frequency(sample, 0.1)
        li = np.random.randint(0, 2, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
        li_onehot = self.one_hot_encoding(li,2)
        aug_1[1-li_onehot[:, 0]] = 0 # the rows are not selected are set as zero.
        aug_2[1 - li_onehot[:, 1]] = 0
        aug_F = aug_1 + aug_2
        return aug_F


    def jitter(self, x, sigma=0.8):
        # https://arxiv.org/pdf/1706.00527.pdf
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

    def remove_frequency(self, x, maskout_ratio=0):
        mask = torch.cuda.FloatTensor(x.shape).uniform_() > maskout_ratio # maskout_ratio are False
        mask = mask.to(x.device)
        return x*mask
        

    def add_frequency(self, x, pertub_ratio=0,):
        mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
        mask = mask.to(x.device)
        max_amplitude = x.max()
        random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
        pertub_matrix = mask*random_am
        return x+pertub_matrix

class NTXentLoss_poly(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss_poly, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = torch.cat((torch.ones(2 * self.batch_size, 1),torch.zeros(2 * self.batch_size, negatives.shape[-1])),dim=-1).to(self.device).long()
        # Add poly loss
        pt = torch.mean(onehot_label* torch.nn.functional.softmax(logits,dim=-1))

        epsilon = self.batch_size
        # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
        loss = CE / (2 * self.batch_size) + epsilon * (1/self.batch_size - pt)
        # loss = CE / (2 * self.batch_size)

        return loss
    

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
