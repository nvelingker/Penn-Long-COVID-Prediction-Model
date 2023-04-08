import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from sklearn.metrics import classification_report,roc_auc_score,recall_score, precision_score, brier_score_loss, average_precision_score, mean_absolute_error
from utils import write_to_pickle, pre_processing_visits, LongCOVIDVisitsDataset2, get_data_min_max, remove_empty_columns, remove_empty_columns_with_non_empty_cls


class create_classifier(nn.Module):
 
    def __init__(self, latent_dim, nhidden=16, N=2, has_static=False, static_input_dim=0):
        super(create_classifier, self).__init__()
        self.gru_rnn = nn.GRU(latent_dim, nhidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, N))

        if has_static:
            self.static_feat = nn.Sequential(
                nn.Linear(static_input_dim, 20),
                nn.ReLU(),
                nn.Linear(20, nhidden))
            self.classifier = nn.Sequential(
                nn.Linear(2*nhidden, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, N))
        
       
    def forward(self, z, static_x=None):
        _, out = self.gru_rnn(z)
        # if static_x is not None:
        if static_x is not None:
            static_feat = self.static_feat(static_x)
            return self.classifier(torch.cat([out.squeeze(0), static_feat], dim = -1))
        else:
            return self.classifier(out.squeeze(0))

# def evaluate_classifier(model, test_loader, dec=None, latent_dim=None, classify_pertp=True, classifier=None,dim=41, device='cuda', reconst=False, num_sample=1):
#     pred = []
#     true = []
#     test_loss = 0
#     for item in test_loader:
#         test_batch, label, person_info_batch = item
#         # train_batch, label, person_info_batch = item
#         if person_info_batch is not None:
#             test_batch, label, person_info_batch = test_batch.float().to(device), label.to(device), person_info_batch.float().to(device)
#         else:
#             test_batch, label = test_batch.float().to(device), label.to(device)
#         batch_len = test_batch.shape[0]
#         observed_data, observed_mask, observed_tp \
#             = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
#         # observed_data = observed_data.float()
#         # observed_mask = observed_mask.float()
#         # observed_tp = observed_tp.float()
#         with torch.no_grad():
#             out = model(
#                 torch.cat((observed_data, observed_mask), 2), observed_tp)
#             if reconst:
#                 qz0_mean, qz0_logvar = out[:, :,
#                                            :latent_dim], out[:, :, latent_dim:]
#                 epsilon = torch.randn(
#                     num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
#                 z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
#                 z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
#                 if classify_pertp:
#                     pred_x = dec(z0, observed_tp[None, :, :].repeat(
#                         num_sample, 1, 1).view(-1, observed_tp.shape[1]))
#                     #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
#                     out = classifier(pred_x, person_info_batch)
#                 else:
#                     out = classifier(z0, person_info_batch)
#             if classify_pertp:
#                 N = label.size(-1)
#                 out = out.view(-1, N)
#                 label = label.view(-1, N)
#                 _, label = label.max(-1)
#                 test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
#             else:
#                 label = label.unsqueeze(0).repeat_interleave(
#                     num_sample, 0).view(-1)
#                 test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * num_sample
#         pred.append(out.cpu())
#         true.append(label.cpu())

    
#     pred = torch.cat(pred, 0)
#     true = torch.cat(true, 0)
#     pred_scores = torch.sigmoid(pred[:, 1])

#     pred = pred.numpy()
#     true = true.numpy()
#     pred_scores = pred_scores.numpy()
#     print("True labels::", true.reshape(-1))
#     print("Predicated labels::", pred_scores.reshape(-1))
#     acc = np.mean(pred.argmax(1) == true)
#     if np.unique(true) <= 1:
#         auc = 0
#     else:
#         auc = roc_auc_score(
#             true, pred_scores) if not classify_pertp else 0.
#     true = true.reshape(-1)
#     pred_labels = (pred_scores > 0.5).reshape(-1).astype(int)
#     recall = recall_score(true.astype(int), pred_labels)
#     precision = precision_score(true.astype(int), pred_labels)
#     print("validation classification Report:\n{}".format(classification_report(true.astype(int), pred_labels)))
#     return test_loss/pred.shape[0], acc, auc, recall, precision, true, pred_labels


def evaluate_classifier(model, test_loader, dec=None, latent_dim=None, classify_pertp=True, classifier=None,dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for item in test_loader:
        test_batch, label, person_info_batch = item
        # train_batch, label, person_info_batch = item
        if person_info_batch is not None:
            test_batch, label, person_info_batch = test_batch.float().to(device), label.to(device), person_info_batch.float().to(device)
        else:
            test_batch, label = test_batch.float().to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
        # observed_data = observed_data.float()
        # observed_mask = observed_mask.float()
        # observed_tp = observed_tp.float()
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                           :latent_dim], out[:, :, latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x, person_info_batch)
                else:
                    out = classifier(z0, person_info_batch)
            if classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * num_sample
        pred.append(out.cpu())
        true.append(label.cpu())

    
    pred = torch.cat(pred, 0)
    true = torch.cat(true, 0)
    pred_scores = torch.sigmoid(pred[:, 1])

    pred = pred.numpy()
    true = true.numpy()
    pred_scores = pred_scores.numpy()
    print("True labels::", true.reshape(-1))
    print("Predicated labels::", pred_scores.reshape(-1))
    acc = np.mean(pred.argmax(1) == true)
    auc = roc_auc_score(
        true, pred_scores) if not classify_pertp else 0.
    true = true.reshape(-1)
    pred_labels = (pred_scores > 0.5).reshape(-1).astype(int)
    recall = recall_score(true.astype(int), pred_labels)
    precision = precision_score(true.astype(int), pred_labels)
    print("validation classification Report:\n{}".format(classification_report(true.astype(int), pred_labels)))
    return test_loss/pred.shape[0], acc, auc, recall, precision, true, pred_labels
def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def compute_losses(dim, observed_data, observed_mask, qz0_mean, qz0_logvar, pred_x, device, norm=True, std=1):
    # observed_data, observed_mask \
    #     = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2*dim]

    noise_std = std  # default 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    if norm:
        logpx /= observed_mask.sum(-1).sum(-1)
        analytic_kl /= observed_mask.sum(-1).sum(-1)
    return logpx, analytic_kl

def train_mTans(lr, norm, std, alpha, k_iwae, dim, latent_dim, rec, dec, classifier, epochs, train_loader, val_loader=None, is_kl=True):
    best_val_loss = float('inf')
    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    # print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    if val_loader is not None:
        val_loss, val_acc, val_auc, val_recall, val_precision,_,_ =  evaluate_classifier(rec, val_loader,latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)
    
        # print("validation performance at epoch::", itr)
        print("validation loss::", val_loss)
        print("validation accuracy::", val_acc)
        print("validation auc score::", val_auc)
        print("validation recall::", val_recall)
        print("validation precision score::", val_precision)

    itr=0
    for itr in range(1,  epochs+ 1):
        print("epoch count::", itr)
        train_recon_loss, train_ce_loss = 0, 0
        mse = 0
        train_n = 0
        train_acc = 0
        #avg_reconst, avg_kl, mse = 0, 0, 0
        if is_kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1-0.99** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1
        local_iter = 0
        for item in train_loader:
            # ((person_info,observed_data, observed_mask, observed_tp), label) = item
            local_iter += 1
            if local_iter % 200 == 0:
                print("local iter::", local_iter, len(train_loader))

            train_batch, label, person_info_batch = item
            if person_info_batch is not None:
                person_info_batch = person_info_batch.float().to(device)
            train_batch = train_batch.float()
            observed_data, observed_mask, observed_tp = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
            observed_data, observed_mask, observed_tp = observed_data.to(device), observed_mask.to(device), observed_tp.to(device)
            # observed_data, observed_mask, observed_tp = observed_data.float(), observed_mask.float(), observed_tp.float()
            label = label.to(device)
            # train_batch, label = train_batch.to(device), label.to(device)
            batch_len  = observed_data.shape[0]
            # observed_data, observed_mask, observed_tp \
            #     = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
            out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
            qz0_mean, qz0_logvar = out[:, :, :latent_dim], out[:, :, latent_dim:]
            epsilon = torch.randn(k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_y = classifier(z0, person_info_batch)
            pred_x = dec(
                z0, observed_tp[None, :, :].repeat(k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
            pred_x = pred_x.view(k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2]) #nsample, batch, seqlen, dim
            # compute loss
            # compute_losses(dim, dec_train_batch, qz0_mean, qz0_logvar, pred_x, device, norm=True, std=1)
            logpx, analytic_kl = compute_losses(
                dim, observed_data, observed_mask, qz0_mean, qz0_logvar, pred_x, device, norm=norm, std=std)
            recon_loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(k_iwae))
            label = label.unsqueeze(0).repeat_interleave(k_iwae, 0).view(-1)
            # print(pred_y.shape, label)
            ce_loss = criterion(pred_y, label.long())
            loss = recon_loss + alpha*ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # train_ce_loss += ce_loss.item() * batch_len
            # train_recon_loss += recon_loss.item() * batch_len
            # train_acc += (pred_y.argmax(1) == label).sum().item()/k_iwae
            # train_n += batch_len
            # mse += mean_squared_error0(observed_data, pred_x.mean(0), 
            #                           observed_mask) * batch_len
        # total_time += time.time() - start_time
        # evaluate_classifier(model, test_loader, dec=None, latent_dim=None, classify_pertp=True, classifier=None,dim=41, device='cuda', reconst=False, num_sample=1)
        train_loader.shuffle = False
        train_loss, train_acc, train_auc, train_recall, train_precision,train_true, train_pred_labels = evaluate_classifier(
            rec, train_loader,latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)
        train_loader.shuffle = True

        if val_loader is not None:
            val_loss, val_acc, val_auc, val_recall, val_precision,true, pred_labels = evaluate_classifier(
                rec, val_loader,latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

            print("validation performance at epoch::", itr)
            print("validation loss::", val_loss)
            print("validation accuracy::", val_acc)
            print("validation auc score::", val_auc)
            print("validation recall::", val_recall)
            print("validation precision score::", val_precision)
            if val_loss <= best_val_loss:
                best_val_loss = min(best_val_loss, val_loss)
                rec_state_dict = rec.state_dict()
                dec_state_dict = dec.state_dict()
                classifier_state_dict = classifier.state_dict()
                optimizer_state_dict = optimizer.state_dict()
                best_true, best_pred_labels = true.copy(), pred_labels.copy()
                best_train_true, best_train_pred_labels = train_true.copy(), train_pred_labels.copy()
                # write_to_pickle(rec_state_dict, "mTans_rec")
                # write_to_pickle(dec_state_dict, "mTans_dec")
                # write_to_pickle(classifier_state_dict, "mTans_classifier")

    return best_true, best_pred_labels, best_train_true, best_train_pred_labels, rec_state_dict, dec_state_dict, classifier_state_dict

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = torch.nn.functional.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)

class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, x, time_steps):
        # time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out

class dec_mtan_rnn(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*nhidden, 2*nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)    
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        
        
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
        
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        # time_steps = time_steps.cpu()
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps).to(self.device)
            key = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            query = self.fixed_time_embedding(time_steps).to(self.device)
            key = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, out)
        out = self.z0_to_obs(out)
        return out        


class mTan_model(nn.Module):
    def __init__(self, rec, dec, classifier, latent_dim, k_iwae, device):
        super(mTan_model, self).__init__()
        self.rec = rec
        self.dec = dec
        self.classifier = classifier
        self.device = device
        self.latent_dim = latent_dim
        self.k_iwae = k_iwae

    def forward(self, *input):
        observed_data, observed_mask, observed_tp, person_info_batch = input
        batch_len  = observed_data.shape[0]
        # observed_data, observed_mask, observed_tp \
        #     = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
        out = self.rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
        qz0_mean, qz0_logvar = out[:, :, :self.latent_dim], out[:, :, self.latent_dim:]
        epsilon = torch.randn(self.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(self.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
        pred_y = self.classifier(z0, person_info_batch)
        # pred_x = self.dec(
        #     z0, observed_tp[None, :, :].repeat(k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
        # pred_x = pred_x.view(k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2]) #nsample, batch, 
        return pred_y


def convert_type(df, all_types):
    column_names = df.columns
    for idx in range(len(column_names)):
        if all_types[idx] == "integer" or all_types[idx] == "short":
            df[column_names[idx]] = df[column_names[idx]].astype('int32')
        elif all_types[idx] == "date":
            df[column_names[idx]] = pd.to_datetime(df[column_names[idx]])
        elif all_types[idx] == "double":
            df[column_names[idx]] = pd.to_numeric(df[column_names[idx]])

    return df


def train_sequential_model_3(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
    print("start")
    # dim=10
    # latent_dim=20
    # rec_hidden=32
    # learn_emb=True
    # enc_num_heads=1
    # num_ref_points=128
    # gen_hidden=30
    # dec_num_heads=1
    # static_input_dim = 10
    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    # lr = 0.0005

    # write_to_pickle(rec.state_dict(), "mTans_rec")
    # write_to_pickle(dec.state_dict(), "mTans_dec")
    # write_to_pickle(classifier.state_dict(), "mTans_classifier")

    # First get the splitted person ids
    train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

    train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = train_person_ids.join(person_information, on="person_id")
    valid_person_info = valid_person_ids.join(person_information, on="person_id")
    # train_person_ids = train_valid_split.loc[train_valid_split["split"] == "train"]
    # valid_person_ids = train_valid_split.loc[train_valid_split["split"] == "valid"]

    # Use it to split the data into training x/y and validation x/y
    # train_recent_visits = train_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")

    # Basic person information
    # train_person_info = train_person_ids.merge(person_information, on="person_id")
    # valid_person_info = valid_person_ids.merge(person_information, on="person_id")
    

    print("start pre-processing!!!")
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both")
    # torch.save(visit_tensor_ls, "train_visit_tensor_ls")
    # torch.save(mask_ls, "train_mask_ls")
    # torch.save(time_step_ls, "train_mask_ls")
    # torch.save(label_tensor_ls, "train_label_tensor_ls")
    # torch.save(person_info_ls, "train_person_info_ls")
    
    # visit_tensor_ls, mask_ls = remove_empty_columns(visit_tensor_ls, mask_ls)

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
    print("finish pre-processing!!!")
    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)
    # train_dataset = LongCOVIDVisitsDataset2(train_person_ids, train_person_info, train_recent_visits, train_labels)
    # valid_dataset = LongCOVIDVisitsDataset2(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

    # Construct dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

    epochs=10
    print(train_dataset.__getitem__(1)[0])
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
    print("data shape::", train_dataset.__getitem__(1)[0].shape)
    print("mask shape::", train_dataset.__getitem__(1)[1].shape)
    print("dim::", dim)
    print(data_min)
    latent_dim=20
    rec_hidden=32
    learn_emb=True
    enc_num_heads=1
    num_ref_points=128
    gen_hidden=30
    dec_num_heads=1
    classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    lr = 0.0005

    

    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)

    best_valid_true, best_valid_pred_labels, best_train_true, best_train_pred_labels, rec_state_dict, dec_state_dict, classifier_state_dict = train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)
    device = torch.device('cpu')
    
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    torch.save(rec_state_dict, os.path.join(root_dir, "model_checkpoints/rec_state_dict"))
    torch.save(dec_state_dict, os.path.join(root_dir, "model_checkpoints/dec_state_dict"))
    torch.save(classifier, os.path.join(root_dir, "model_checkpoints/classifier"))
    
    
    # rec.load_state_dict(rec_state_dict)
    # dec.load_state_dict(dec_state_dict)
    # classifier.load_state_dict(classifier_state_dict)
    # # write_to_pickle(rec_state_dict, "mTans_rec")
    # # write_to_pickle(dec_state_dict, "mTans_dec")
    # # write_to_pickle(classifier_state_dict, "mTans_classifier")
    # # read_from_pickle()
    # print("save models successfully")
    # return rec, dec, classifier

# def train_sequential_model_3(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
# def train_sequential_model_0():
#     print("start")

#     # # First get the splitted person ids
#     # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
#     # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

#     # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
#     # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
#     # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
#     # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
#     # # print(train_recent_visits.show())

#     # train_person_info = train_person_ids.join(person_information, on="person_id")
#     # valid_person_info = valid_person_ids.join(person_information, on="person_id")

#     train_recent_visits = pd.read_csv("/home/wuyinjun/train_recent_visits.csv")
#     all_types = ["string","string","date","date","date","short","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","double","integer","date"]
#     train_recent_visits = convert_type(train_recent_visits, all_types)

#     train_person_ids = pd.read_csv("/home/wuyinjun/train_person_ids.csv")
#     all_types = ["string", "string"]
#     train_person_ids = convert_type(train_person_ids, all_types)

#     train_labels = pd.read_csv("/home/wuyinjun/train_labels.csv")
#     all_types = ["string","string","date","integer","integer","double"]
#     train_labels = convert_type(train_labels, all_types)


#     train_person_info = pd.read_csv("/home/wuyinjun/train_person_info.csv")
#     all_types = ["string","string","double","double","long","long","long"]
#     train_person_info = convert_type(train_person_info, all_types)


#     # train_person_info = train_person_info.set_index("person_id")
#     # train_recent_visits = train_recent_visits.set_index(["person_id", "visit_date"])
#     # train_labels = train_labels.set_index("person_id")




#     # train_person_ids = train_valid_split.loc[train_valid_split["split"] == "train"]
#     # valid_person_ids = train_valid_split.loc[train_valid_split["split"] == "valid"]

#     # Use it to split the data into training x/y and validation x/y
#     # train_recent_visits = train_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
#     # valid_recent_visits = valid_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
#     # train_labels = train_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")
#     # valid_labels = valid_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")

#     # Basic person information
#     # train_person_info = train_person_ids.merge(person_information, on="person_id")
#     # valid_person_info = valid_person_ids.merge(person_information, on="person_id")
    

#     print("start pre-processing!!!")
#     visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids, train_person_info, train_recent_visits, train_labels, setup="both")
#     # visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both")
#     # torch.save(visit_tensor_ls, "train_visit_tensor_ls")
#     # torch.save(mask_ls, "train_mask_ls")
#     # torch.save(time_step_ls, "train_mask_ls")
#     # torch.save(label_tensor_ls, "train_label_tensor_ls")
#     # torch.save(person_info_ls, "train_person_info_ls")
    
#     # visit_tensor_ls, mask_ls = remove_empty_columns(visit_tensor_ls, mask_ls)

#     # valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
#     print("finish pre-processing!!!")
#     visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
#     # valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

#     data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

#     train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

#     # valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)
#     # train_dataset = LongCOVIDVisitsDataset2(train_person_ids, train_person_info, train_recent_visits, train_labels)
#     # valid_dataset = LongCOVIDVisitsDataset2(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

#     # Construct dataloaders
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
#     # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

#     epochs=10
#     print(train_dataset.__getitem__(1)[0])
#     dim = train_dataset.__getitem__(1)[0].shape[-1]
#     static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
#     print("data shape::", train_dataset.__getitem__(1)[0].shape)
#     print("mask shape::", train_dataset.__getitem__(1)[1].shape)
#     print("dim::", dim)
#     print(data_min)
#     latent_dim=20
#     rec_hidden=32
#     learn_emb=True
#     enc_num_heads=1
#     num_ref_points=128
#     gen_hidden=30
#     dec_num_heads=1
#     classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
#     device = torch.device(
#         'cuda' if torch.cuda.is_available() else 'cpu')
#     print("device::", device)
#     rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
#     dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
#     lr = 0.0005

    

#     rec = rec.to(device)
#     dec = dec.to(device)
#     classifier = classifier.to(device)

#     best_valid_true, best_valid_pred_labels, best_train_true, best_train_pred_labels, rec_state_dict, dec_state_dict, classifier_state_dict = train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, None, is_kl=True)
#     device = torch.device('cpu')
#     write_to_pickle(rec_state_dict, "mTans_rec")
#     write_to_pickle(dec_state_dict, "mTans_dec")
#     write_to_pickle(classifier_state_dict, "mTans_classifier")
#     # read_from_pickle()
#     print("save models successfully")

# if __name__ == "__main__":
#     train_sequential_model_3()