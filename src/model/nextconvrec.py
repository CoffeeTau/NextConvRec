import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention

from model.session_gnn import SessionGNN

class NextConvRecModel(SequentialRecModel):
    def __init__(self, args):
        super(NextConvRecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = NextConvRecEncoder(args)
        self.session_gnn = SessionGNN(args.hidden_size)
        if hasattr(args, "pretrained_item_emb") and args.pretrained_item_emb is not None:
            self.item_embeddings = nn.Embedding.from_pretrained(
                args.pretrained_item_emb, freeze=False
            )
            print("[INFO] item_embeddings initialized from GCN.")
        else:
            self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size)
            print("[INFO] item_embeddings randomly initialized.")

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        mask = input_ids > 0
        raw_emb = self.item_embeddings(input_ids)
        B, L = input_ids.shape
        A = torch.zeros(B, L, L, device=input_ids.device)
        src = torch.arange(L - 1, device=input_ids.device).unsqueeze(0).expand(B, -1)
        dst = src + 1
        mask = (input_ids[:, :-1] > 0) & (input_ids[:, 1:] > 0)
        A[torch.arange(B).unsqueeze(1), src, dst] = mask.float()
        gnn_emb = self.session_gnn(raw_emb, A)
        sequence_emb = self.LayerNorm(gnn_emb + raw_emb)
        sequence_emb = self.dropout(sequence_emb)
        sequence_emb = self.LayerNorm(gnn_emb)
        sequence_emb = self.dropout(sequence_emb)
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_emb = self.position_embeddings(position_ids)
        sequence_emb += position_emb
        item_encoded_layers = self.item_encoder(
            sequence_emb,
            mask,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )

        if all_sequence_output:
            return item_encoded_layers
        else:
            return item_encoded_layers[-1]
    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.T)
        loss_main = F.cross_entropy(logits, answers)
        gcn_item_emb = self.item_embeddings.weight.detach()
        pos = gcn_item_emb[answers]
        neg = gcn_item_emb[neg_answers]
        pos_sim = F.cosine_similarity(seq_output, pos, dim=-1)
        neg_sim = F.cosine_similarity(seq_output, neg, dim=-1)
        margin = 0.2
        contrastive_loss = F.relu(margin + neg_sim - pos_sim).mean()
        total_loss = loss_main + 0.05 * contrastive_loss

        return total_loss
class NextConvRecEncoder(nn.Module):
    def __init__(self, args):
        super(NextConvRecEncoder, self).__init__()
        self.args = args
        block = NextConvRecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, mask, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, mask, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
class NextConvRecBlock(nn.Module):
    def __init__(self, args):
        super(NextConvRecBlock, self).__init__()
        self.layer = NextConvRecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, mask, attention_mask):
        layer_output = self.layer(hidden_states, mask, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=1, groups=groups,bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel,
                                            stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,bias=False)


    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)

        return out



class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):

        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel
    def forward(self,x):

        input = x
        B, M, D, N = x.shape
        x = x.reshape(B,M*D,N)
        x = self.dw(x)
        x = x.reshape(B,M,D,N)
        x = x.reshape(B*M,D,N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = int(dmodel * ffn_ratio)
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x
class NextConvRecLayer(nn.Module):
    def __init__(self, args):
        super(NextConvRecLayer, self).__init__()
        self.args = args
        self.filter_layer = FrequencyLayer(args)
        self.attention_layer = MultiHeadAttention(args)
        self.alpha = args.alpha
        self.Modern_tcn = Stage(ffn_ratio=2, num_blocks=1, large_size=25, small_size=5, dmodel=64, dw_model=64, nvars=1)




    def forward(self, input_tensor, mask, attention_mask):
        dsp = self.filter_layer(input_tensor)

        input_tensor = input_tensor.unsqueeze(1)
        input_tensor = input_tensor.permute(0, 1, 3, 2)
        gsp = self.Modern_tcn(input_tensor)
        gsp = gsp.squeeze(1)
        gsp = gsp.permute(0, 2, 1)
        hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * gsp


        return hidden_states
class FrequencyLayer(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.c = args.c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')


        
        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass



        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
