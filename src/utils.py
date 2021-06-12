import time
import numpy as np
from sklearn.metrics import f1_score
from models import *


class trainer_model(object):
    def __init__(self,
                 ds,
                 model,
                 p_train,
                 p_val,
                 post,
                 history,
                 idx,
                 node_feats,
                 adjs,
                 batch_size,
                 epochs,
                 weight_decay,
                 model_path,
                 class_weight,
                 early_stopping,
                 device,
                 lr=0.01):
        self.ds = ds
        self.model = model
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight).to(device))
        self.lr = lr
        self.p_train = p_train
        self.p_val = p_val
        self.post = post
        self.history = history
        self.idx = idx
        self.node_feats = node_feats
        self.adjs = adjs
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path
        self.early_stopping = early_stopping
        self.device = device
        self.data_time = 0

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def train(self):
        best_val_loss = float("inf")
        best_val_f1 = 0.
        last_step = 0
        step = 0

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.train_epoch(epoch)
            val_loss, val_f1 = self.valid_epoch()
            print('| end of epoch {} | time: {:5.2f}s | valid classification loss {:5.4f} | valid classification '
                  'f1 {:5.4f} | data loader time: {:5.2f}s'.format(epoch, time.time() - epoch_start_time, val_loss, val_f1, self.data_time))
            self.data_time = 0

            step += 1
            if best_val_f1 < val_f1:
                best_val_f1 = val_f1
                # print(best_val_loss)
                last_step = step
                torch.save(self.model.state_dict(), self.model_path + self.ds + '.pth')
            else:
                if step - last_step >= self.early_stopping:
                    print('\nearly stop by {} epochs, val loss: {:.4f}'.format(self.early_stopping, best_val_loss))
                    return

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.
        total_f1 = []
        start_time = time.time()

        for batch, i in enumerate(range(0, len(self.p_train), self.batch_size)):

            self.optimizer.zero_grad()
            d_time = time.time()
            u_comments, u_mask, a_comments, a_mask, pf, label, uidx, aidx = self.get_batch(self.p_train,
                                                                                           i,
                                                                                           self.batch_size,
                                                                                           self.history,
                                                                                           self.post,
                                                                                           self.idx)
            self.data_time += time.time() - d_time

            pf = pf.unsqueeze(1)
            pred, _ = self.model(u_comments.permute(1, 0, 2).float(),
                              a_comments.permute(1, 0, 2).float(),
                              pf.permute(1, 0, 2).float(),
                              u_mask,
                              a_mask,
                              self.node_feats,
                              self.adjs,
                              uidx,
                              aidx
                              )

            loss = self.criterion(pred, label.long())
            _, pred_label = torch.max(pred, 1)

            total_f1.append(f1_score(label.to('cpu').numpy(), pred_label.to('cpu').numpy()))

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            log_interval = 50
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                cur_f1 = np.mean(total_f1)
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:3d}/{:3d} batches | '
                      'lr {:5.6f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:5.2f} | f1 {:5.2f} |'.format(
                    epoch, batch, len(self.p_train) // self.batch_size, self.optimizer.param_groups[0]['lr'],
                                  elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss), cur_f1))

                total_loss = 0
                total_f1 = []
                start_time = time.time()

    def valid_epoch(self):
        self.model.eval()
        total_loss = 0.
        all_f1 = []
        with torch.no_grad():
            for batch, i in enumerate(range(0, len(self.p_val), self.batch_size)):
                d_time = time.time()
                u_comments, u_mask, a_comments, a_mask, pf, label, uidx, aidx = self.get_batch(self.p_val,
                                                                                               i,
                                                                                               self.batch_size,
                                                                                               self.history,
                                                                                               self.post,
                                                                                               self.idx)
                self.data_time += time.time() - d_time
                pf = pf.unsqueeze(1)
                pred, _ = self.model(u_comments.permute(1, 0, 2).float(),
                                     a_comments.permute(1, 0, 2).float(),
                                  pf.permute(1, 0, 2).float(),
                                  u_mask,
                                  a_mask,
                                  self.node_feats,
                                  self.adjs,
                                  uidx,
                                  aidx
                                  )
                loss = self.criterion(pred, label.long())
                total_loss += loss.item()

                _, pred_label = torch.max(pred, 1)
                all_f1.append(f1_score(label.to('cpu').numpy(), pred_label.to('cpu').numpy()))

        return total_loss / len(all_f1), np.mean(all_f1)

    def get_batch(self, pairs, i, batch_size, input_feature, post_dec, idx):
        seq_len = min(batch_size, len(pairs) - i)
        pairs_batch = pairs[i: i + seq_len, :]
        label = pairs[i: i + seq_len, 3].astype(int)

        uidx = np.array([idx.index(u) for u in pairs_batch[:, 1]])
        aidx = np.array([idx.index(a) for a in pairs_batch[:, 2]])

        pf = []
        uc = []
        ac = []

        u_seq_len = 0
        u_len = []
        a_seq_len = 0
        a_len = []

        for row in range(pairs_batch.shape[0]):

            try:
                pf_elem = post_dec.loc[post_dec['P_id'] == pairs_batch[row][0], 'Feature'].tolist()[0]
            except:
                pf_elem = np.random.rand(768)
            pf.append(pf_elem)

            try:
                uc_elem = input_feature.loc[input_feature['Ori_user_id'] == pairs_batch[row][1], 'Feature'].tolist()
            except:
                uc_elem = [np.random.rand(768)]
            if len(uc_elem) > u_seq_len:
                u_seq_len = len(uc_elem)
            uc.append(uc_elem)
            u_len.append(len(uc_elem))

            try:
                ac_elem = input_feature.loc[input_feature['Ori_user_id'] == pairs_batch[row][2], 'Feature'].tolist()
            except:
                ac_elem = [np.random.rand(768)]
            if len(ac_elem) > a_seq_len:
                a_seq_len = len(ac_elem)
            ac.append(ac_elem)
            a_len.append(len(ac_elem))

        u_comments = np.zeros((seq_len, u_seq_len, 768))
        u_mask = np.ones((seq_len, u_seq_len))
        a_comments = np.zeros((seq_len, a_seq_len, 768))
        a_mask = np.ones((seq_len, a_seq_len))

        for i in range(0, seq_len):
            u_comments[i, 0:len(uc[i])] = uc[i]
            u_mask[i, 0:u_len[i]] = 0
            a_comments[i, 0:len(ac[i])] = ac[i]
            a_mask[i, 0:a_len[i]] = 0

        return torch.tensor(u_comments).to(self.device), \
               torch.tensor(u_mask).bool().to(self.device), \
               torch.tensor(a_comments).to(self.device), \
               torch.tensor(a_mask).bool().to(self.device), \
               torch.tensor(pf).to(self.device), \
               torch.tensor(label).to(self.device), \
               torch.tensor(uidx).long().to(self.device), \
               torch.tensor(aidx).long().to(self.device)
