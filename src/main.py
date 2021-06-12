import random
import argparse
from data_center import DataCenter
from utils import *
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

parser = argparse.ArgumentParser(description='pytorch version of GFCI')

parser.add_argument('--dataset', type=str, default='baby')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--b_sz', type=int, default=128)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--win_size', type=str, default='month')
parser.add_argument('--fusion_dim', type=int, default=768)
parser.add_argument('--fusion_factor', type=int, default=4)
parser.add_argument('--train_proportion', type=float, default=0.8)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--gcn_dim', type=int, default=128)
parser.add_argument('--cuda_id', type=str, default='0')
parser.add_argument('--model_path', type=str, default='./model_')
args = parser.parse_args()

device = torch.device("cuda:" + args.cuda_id if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    print('*' * 20 + 'Loading data' + '*' * 20)
    ds = args.dataset
    print('ds: ', ds)
    dCenter = DataCenter(device)
    dCenter.load_dataSet(args.dataset, args.win_size, args.train_proportion)

    print('*' * 20 +'Initializing model' + '*' * 20)
    model = Model(conv_num=1,
                  k=args.k,
                  hid_dim=32,
                  n_heads=2,
                  pf_dim=64,
                  n_layers=1,
                  gcn_in_dim=getattr(dCenter, ds + '_feature').shape[1],
                  gcn_dim=args.gcn_dim,
                  fusion_dim=args.fusion_dim,
                  fusion_factor=args.fusion_factor,
                  gcn_hops=len(getattr(dCenter, ds + '_adj_windows')))

    model = model.float()
    model.apply(initialize_weights)
    model.to(device)

    print('*' * 20 + 'Training model' + '*' * 20)
    lb, lcounts = np.unique(getattr(dCenter, ds + '_train')[:, 3], return_counts=True)
    trainer = trainer_model(ds=args.dataset,
                            model=model,
                            p_train=getattr(dCenter, ds + '_train'),
                            p_val=getattr(dCenter, ds + '_val'),
                            post=getattr(dCenter, ds + '_post'),
                            history=getattr(dCenter, ds + '_history'),
                            idx=getattr(dCenter, ds + '_idx'),
                            node_feats=getattr(dCenter, ds + '_feature'),
                            adjs=getattr(dCenter, ds + '_adj_windows'),
                            batch_size=args.b_sz,
                            epochs=args.epochs,
                            weight_decay=1e-4,
                            model_path=args.model_path,
                            class_weight=[1.0, float(lcounts[0]/lcounts[1])],
                            early_stopping=5,
                            device=device,
                            lr=5e-4)

    trainer.train()

    print('*' * 20 + 'Testing model' + '*' * 20)
    model.load_state_dict(torch.load(args.model_path + args.dataset + '.pth'))
    model.eval()

    all_acc = []
    all_prec = []
    all_recall = []
    all_f1 = []

    with torch.no_grad():
        for batch, i in enumerate(range(0, getattr(dCenter, ds + '_test').shape[0], args.b_sz)):
            u_comments, u_mask, a_comments, a_mask, pf, label, uidx, aidx = trainer.get_batch(getattr(dCenter, ds + '_test'),
                                                                                              i,
                                                                                              args.b_sz,
                                                                                              getattr(dCenter, ds + '_history'),
                                                                                              getattr(dCenter, ds + '_post'),
                                                                                              getattr(dCenter, ds + '_idx'))
            pf = pf.unsqueeze(1)
            pred, nodes_feats = model(u_comments.permute(1, 0, 2).float(),
                                     a_comments.permute(1, 0, 2).float(),
                                     pf.permute(1, 0, 2).float(),
                                     u_mask,
                                     a_mask,
                                     getattr(dCenter, ds + '_feature'),
                                     getattr(dCenter, ds + '_adj_windows'),
                                     uidx,
                                     aidx)
            np.save('nodes_feats_' + args.dataset, nodes_feats.to('cpu').numpy())
            _, pred_label = torch.max(pred, 1)

            all_acc.append(accuracy_score(label.to('cpu').numpy(), pred_label.to('cpu').numpy()))
            all_prec.append(precision_score(label.to('cpu').numpy(), pred_label.to('cpu').numpy()))
            all_recall.append(recall_score(label.to('cpu').numpy(), pred_label.to('cpu').numpy()))
            all_f1.append(f1_score(label.to('cpu').numpy(), pred_label.to('cpu').numpy()))

    np.save('idx_' + args.dataset, getattr(dCenter, ds + '_idx'))
    print('testing acc：', np.mean(all_acc))
    print('testing prec：', np.mean(all_prec))
    print('testing recall：', np.mean(all_recall))
    print('testing f1：', np.mean(all_f1))

    with open('./result.txt', 'a') as f:
        f.write(str(args) + '\n')
        f.write('testing acc：' + str(np.mean(all_acc)) + '\n')
        f.write('testing prec：' + str(np.mean(all_prec)) + '\n')
        f.write('testing recall：' + str(np.mean(all_recall)) + '\n')
        f.write('testing f1：' + str(np.mean(all_f1)) + '\n')
