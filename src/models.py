from torch.nn import MultiheadAttention
import torch.nn.functional as F
import torch.nn as nn
import torch
import math


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.bn = nn.BatchNorm1d(pf_dim)

    def forward(self, x):
        x = self.bn(torch.relu(self.fc_1(x)).permute(1, 2, 0)).permute(2, 0, 1)
        x = self.fc_2(x)

        return x


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * (1 - mask) + mask * (-1e30)


class U2AAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        w4U = torch.empty(d_model, 1)
        w4A = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4U)
        nn.init.xavier_uniform_(w4A)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4U)
        self.w4Q = nn.Parameter(w4A)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, U, A, Amask):

        U = U.transpose(0, 1)
        A = A.transpose(0, 1)
        batch_size_c = U.size()[0]

        batch_size, Lq, d_model = A.shape
        S = self.trilinear_for_attention(U, A)

        Amask = Amask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Amask), dim=2)

        out = torch.bmm(S1, A)

        return out.transpose(0, 1), S1

    def trilinear_for_attention(self, U, A):
        batch_size, Lc, d_model = U.shape
        batch_size, Lq, d_model = A.shape

        subres0 = torch.matmul(U, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(A, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(U * self.w4mlu, A.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class SelfAttnLayer(nn.Module):
    def __init__(self,
                 conv_num,
                 k,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout=0.2):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv1d(768, hid_dim, k, padding=k // 2) for _ in range(conv_num)])
        self.bn_C = nn.ModuleList([nn.BatchNorm1d(hid_dim) for _ in range(conv_num)])

        self.self_attention = MultiheadAttention(hid_dim, n_heads)
        self.layer_norm_attn = nn.LayerNorm(hid_dim)
        self.bn_attn = nn.BatchNorm1d(hid_dim)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim)
        self.layer_norm_ff = nn.LayerNorm(hid_dim)
        self.bn_ff = nn.BatchNorm1d(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        for i, conv in enumerate(self.convs):
            _src = conv(src.permute(1, 2, 0)).permute(2, 0, 1)
            src = self.bn_C[i](_src.permute(1, 2, 0)).permute(2, 0, 1)
            src = self.dropout(src)

        _src, _ = self.self_attention(src, src, src, key_padding_mask=src_mask)
        src = self.layer_norm_attn(src + self.bn_attn(_src.permute(1, 2, 0)).permute(2, 0, 1))
        src = self.dropout(src)

        _src = self.positionwise_feedforward(src)
        src = self.layer_norm_ff(src + self.bn_ff(_src.permute(1, 2, 0)).permute(2, 0, 1))
        src = self.dropout(src)

        return src


class UserTopicAttn(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout=0.2):
        super().__init__()

        self.self_attention = MultiheadAttention(hid_dim, n_heads)
        self.layer_norm_attn = nn.LayerNorm(hid_dim)
        self.bn_attn = nn.BatchNorm1d(hid_dim)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim)
        self.layer_norm_ff = nn.LayerNorm(hid_dim)
        self.bn_ff = nn.BatchNorm1d(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, topic, user, user_mask):

        _topic, _ = self.self_attention(topic, user, user, key_padding_mask=user_mask)
        topic = self.layer_norm_attn(topic + self.bn_attn(_topic.permute(1, 2, 0)).permute(2, 0, 1))
        topic = self.dropout(topic)

        _topic = self.positionwise_feedforward(topic)
        topic = self.layer_norm_ff(topic + self.bn_ff(_topic.permute(1, 2, 0)).permute(2, 0, 1))
        topic = self.dropout(topic)

        return topic


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Gate(nn.Module):
    def __init__(self, nfeat_gcn, init_gcn):
        super(Gate, self).__init__()

        self.weight_uz = nn.Linear(init_gcn, nfeat_gcn)
        self.weight_wz = nn.Linear(nfeat_gcn, nfeat_gcn)
        self.weight_uh = nn.Linear(init_gcn, nfeat_gcn)
        self.weight_wh = nn.Linear(nfeat_gcn, nfeat_gcn)

    def forward(self, last, current):
        z = self.weight_wz(current) + self.weight_uz(last)
        z = torch.sigmoid(z)

        output = (1 - z) * self.weight_wh(current) + z * self.weight_uh(last)
        output = torch.tanh(output)

        return output


class FusionLayer(nn.Module):
    def __init__(self,
                 interest_dim,
                 relation_dim,
                 fusion_dim,
                 factor_num,
                 dropout=0.2):
        super().__init__()

        self.weight_i = nn.Linear(interest_dim, fusion_dim)
        self.weight_r = nn.Linear(relation_dim, fusion_dim)
        self.fc = nn.Linear(int(fusion_dim/factor_num), 2)
        self.factor_num = factor_num

        self.dropout = nn.Dropout(dropout)

    def forward(self, interest, relationship):
        interest = self.weight_i(interest)
        relationship = self.weight_r(relationship)
        output = torch.mul(interest, relationship)
        output = self.dropout(output)
        output = output.view(output.shape[0], 1, int(output.shape[1]/self.factor_num), self.factor_num)
        output = torch.mean(output, dim=3, keepdim=True)
        output = output.squeeze()
        output = torch.sqrt(F.relu(output)) - torch.sqrt(F.relu(-output))
        output = F.normalize(output)
        output = self.fc(output)

        return output


class Model(nn.Module):
    def __init__(self,
                 conv_num,
                 k,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 n_layers,
                 gcn_in_dim,
                 gcn_dim,
                 gcn_hops,
                 fusion_dim,
                 fusion_factor,
                 dropout=0.2):
        super().__init__()

        self.user_layers = nn.ModuleList([SelfAttnLayer(conv_num,
                                                        k,
                                                        hid_dim,
                                                        n_heads,
                                                        pf_dim)
                                          for _ in range(n_layers)])

        self.author_layers = nn.ModuleList([SelfAttnLayer(conv_num,
                                                          k,
                                                          hid_dim,
                                                          n_heads,
                                                          pf_dim)
                                            for _ in range(n_layers)])

        self.post_layers = nn.ModuleList([SelfAttnLayer(conv_num,
                                                          k,
                                                          hid_dim,
                                                          n_heads,
                                                          pf_dim)
                                            for _ in range(n_layers)])

        self.user2author_attn = U2AAttention(hid_dim)
        self.u2a_bn = nn.BatchNorm1d(hid_dim)

        self.ut_attention = UserTopicAttn(hid_dim, n_heads, pf_dim)
        self.ut_bn = nn.BatchNorm1d(hid_dim)
        self.at_attention = UserTopicAttn(hid_dim, n_heads, pf_dim)
        self.at_bn = nn.BatchNorm1d(hid_dim)

        self.gcn_hops = gcn_hops

        self.gcns = nn.ModuleList(
            [GraphConvolution(gcn_in_dim, gcn_dim) if i == 0 else GraphConvolution(gcn_dim, gcn_dim) for i in range(gcn_hops)])
        self.bn_gcns = nn.ModuleList([nn.BatchNorm1d(gcn_dim) for _ in range(gcn_hops)])
        self.gates = nn.ModuleList([Gate(gcn_dim, gcn_dim) for _ in range(gcn_hops - 1)])

        self.dropout = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim * 3, 2)

        self.fusion = FusionLayer(hid_dim * 3, gcn_dim * 2, fusion_dim, fusion_factor)

    def forward(self, user, author, topic, user_mask, author_mask, nodes_feats, adjs, idx_u, idx_a):

        for layer in self.user_layers:
            user = layer(user, user_mask)

        for layer in self.author_layers:
            author = layer(author, author_mask)

        for layer in self.post_layers:
            topic = layer(topic, None)

        author, attn1 = self.user2author_attn(user, author, author_mask)
        author = self.u2a_bn(author.permute(0, 2, 1)).permute(0, 2, 1)

        output1 = self.ut_attention(topic, user, user_mask)
        output1 = self.ut_bn(output1.squeeze())
        output2 = self.at_attention(topic, author, author_mask)
        output2 = self.at_bn(output2.squeeze())

        for i, hop in enumerate(self.gcns):
            last_nodes_feats = nodes_feats
            nodes_feats = self.bn_gcns[i](self.gcns[i](nodes_feats, adjs[i]))

            if i > 0:
                nodes_feats = F.relu(nodes_feats)
                nodes_feats = self.dropout(nodes_feats)
                nodes_feats = self.gates[i-1](last_nodes_feats, nodes_feats)

        nodes_feats = F.relu(nodes_feats)
        output = self.fusion(torch.cat([output1.squeeze(), output2.squeeze(), topic.squeeze()], dim=-1),
                             torch.cat([nodes_feats[idx_u], nodes_feats[idx_a]], dim=-1))

        return output, nodes_feats