# updated from https://github.com/facebookresearch/moco
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

import model.ResNet as models

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, model='resnet18',dim=512, K=65536, m=0.999, T=0.2, pretrained=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))
        self.encoder_q = getattr(models, model)(
            pretrained=pretrained)  # Query Encoder

        self.encoder_k = getattr(models, model)(
           pretrained=pretrained)  # Key Encoder

        if model == 'resnet18' or model == 'resnet34':
            n_channels = 512
        elif model == 'resnet50' or model == 'resnet101' or model == 'resnet152':
            n_channels = 2048
        else:
            raise NotImplementedError('model not supported: {}'.format(model))
        self.feature_length=n_channels
        self.encoder_q.fc = projection_MLP(n_channels,dim)
        self.encoder_k.fc = projection_MLP(n_channels,dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)#normalize across queue instead of each example

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        #assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr+batch_size<self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            end=self.K
            self.queue[:, ptr:] = keys[:end-ptr].T
            self.queue[:,:ptr + batch_size-end]=keys[end-ptr:].T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def shuffled_idx(self, batch_size):
        '''
        Generation of the shuffled indexes for the implementation of ShuffleBN.
        https://github.com/HobbitLong/CMC.
        args:
            batch_size (Tensor.int()):  Number of samples in a batch
        returns:
            shuffled_idxs (Tensor.long()): A random permutation index order for the shuffling of the current minibatch
            reverse_idxs (Tensor.long()): A reverse of the random permutation index order for the shuffling of the
                                            current minibatch to get back original sample order
        '''

        # Generate shuffled indexes
        shuffled_idxs = torch.randperm(batch_size).long().cuda()

        reverse_idxs = torch.zeros(batch_size).long().cuda()

        value = torch.arange(batch_size).long().cuda()

        reverse_idxs.index_copy_(0, shuffled_idxs, value)

        return shuffled_idxs, reverse_idxs
    # @torch.no_grad()
    # def _batch_shuffle_ddp(self, x):
    #     """
    #     Batch shuffle, for making use of BatchNorm.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # random shuffle index
    #     idx_shuffle = torch.randperm(batch_size_all).cuda()
    #
    #     # broadcast to all gpus
    #     torch.distributed.broadcast(idx_shuffle, src=0)
    #
    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)
    #
    #     # shuffled index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this], idx_unshuffle
    #
    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     """
    #     Undo batch shuffle.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this]
    def Calculate_qfeature(self,im_q):
        q=self.encoder_q(im_q,use_feature=True)
        return q
    def Calculate_kfeature(self,im_k):
        k=self.encoder_k(im_k,use_feature=True)
        return k
    def forward(self, im_q, im_k,type=0):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        batch_size = q.size(0)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size)
            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            im_k = im_k[shuffled_idxs]
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k = k[reverse_idxs]
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        if type==0:
            return logits, labels
        elif type==1:
            with torch.no_grad():
                k=self.encoder_k(im_k, use_feature=True)
            return k,logits,labels





    # def __getattr__(self, name):
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         return getattr(self.model.module, name)

# utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#
#     output = torch.cat(tensors_gather, dim=0)
#     return output


import torch.nn as nn



''' SimCLR Projection Head '''


class projection_MLP(nn.Module):
    def __init__(self, n_channels,out_channels):
        '''Projection head for the pretraining of the resnet encoder.
            - Uses the dataset and model size to determine encoder output
                representation dimension.
            - Outputs to a dimension of 128, and uses non-linear activation
                as described in SimCLR paper: https://arxiv.org/pdf/2002.05709.pdf
        '''
        super(projection_MLP, self).__init__()


        self.projection_head = nn.Sequential()
        self.projection_head.add_module('W1', nn.Linear(
            n_channels, n_channels))
        self.projection_head.add_module('ReLU', nn.ReLU())
        self.projection_head.add_module('W2', nn.Linear(
            n_channels, out_channels))

    def forward(self, x):
        return self.projection_head(x)
