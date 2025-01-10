import torch
import torch.nn as nn
import sys

sys.path.append("..")

from prompts.imagenet_template import openai_imagenet_template, sub_imagenet_template

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from mmseg.registry import MODELS

from torchvision import transforms
import torch.nn.functional as F
from einops import rearrange

from open_clip import create_model, tokenizer
from myutils import UnNormalize


@MODELS.register_module()
class ProxyCLIPSegmentation(BaseSegmentor):
    def __init__(self, clip_type, model_type, vfm_model, name_path, checkpoint=None, device=torch.device('cuda'),
                 prob_thd=0.0, logit_scale=40, beta=1.2, gamma=3.0, slide_stride=112, slide_crop=336):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        #torch.save(self.clip, "clip_model_full.pth")
        # 打印模型的 Transformer 结构
        #print(self.clip.visual)  # 打印视觉模型结构
        #print(self.clip.visual.transformer)  # Transformer 结构
        #print(self.clip.visual.transformer.resblocks[-1])  # 最后一个 Transformer block
        #print(self.clip.visual.transformer.resblocks[-1].attn)  # Attention 模块

        self.clip.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.vfm_model = vfm_model

        if vfm_model == 'dino':
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        elif vfm_model == 'unet':
            #self.vfm = torch.hub.load('milesial/Pytorch-UNet:master', 'unet_carvana', pretrained=True, scale=0.5)
            self.vfm = torch.hub.load('milesial/Pytorch-UNet:master', 'unet_carvana', pretrained=True, scale=1)
        else:
            print("vlm_model not supported")

        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(device)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.clip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()
        #print(self.query_features)
        #torch.save(self.query_features,'query_features.pt')
        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.beta = beta
        self.gamma = gamma

    @torch.no_grad()
    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]

        clip_token_size = img.shape[-2] // self.clip.visual.patch_size[0], img.shape[-1] // self.clip.visual.patch_size[1]

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)

        imgs_norm = imgs_norm.half()
        #print(imgs_norm.shape)
        if self.vfm_model == 'dino':
            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)

            # Forward pass in the model
            feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
            print("feat",feat.shape)
            nb_im = feat.shape[0]  # Batch size
            #print("nb_im",nb_im)
            nb_tokens = feat.shape[1]  # Number of tokens
            #print("nb_tokens",nb_tokens)
            nh = self.vfm.blocks[0].attn.num_heads  # Number of heads

            qkv = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4)
            )
            #print("qkv",feat_out["qkv"].shape)
            q, k, v = qkv[0], qkv[1], qkv[2]
            #print("q",q.shape)
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
            q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
            v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
            #print("q1",q.shape)
            #attn_scores = torch.matmul(q, k.transpose(-1, -2))
            #attn_weights = F.softmax(attn_scores, dim=-1)
            #print("a",attn_weights.shape,attn_scores.shape)
            #new_features = torch.matmul(attn_weights, v) 
            
            #print("new_feature:",new_features.shape)
            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-2] // patch_size
            print("patch_size",patch_size)
            print(imgs_norm[0].shape[-2])
            #ex_feats = new_features.reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
            # print(q)
            # ex_feats = q.reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
            # ex_feats = k.reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
            # ex_feats = v.reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
            #torch.save({'q': q, 'k': k, 'v': v}, 'qkv.pt')
            
            ex_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
            print("ex_feats",ex_feats.shape)
        elif self.vfm_model == 'unet':
            # 获取中间层特征
            features = self.vfm.get_intermediate_layers(imgs_norm)
            # 提取第九块（例如 up1）的特征
            up1_feature = features['outc']
            patch_size=8
            nb_im = up1_feature.shape[0]
            I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-2] // patch_size
            ex_feats = up1_feature[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
        else:
            I, J = clip_token_size
            ex_feats = None
        '''
        image_features = self.vfm(imgs_norm).reshape(-1, I, J)
        print("image_features",image_features.shape)
        '''
        image_features = self.clip.encode_image(img.half(),
                                               external_feats=ex_feats,
                                               beta=self.beta,
                                               gamma=self.gamma)
        
        '''
        dino_dim = ex_feats.size(-1)
        clip_dim = self.query_features.size(-1)

        if dino_dim != clip_dim:
            projector = nn.Linear(dino_dim, clip_dim).to("cuda:0")
            ex_feats = ex_feats.to(projector.weight.dtype)
            ex_feats = projector(ex_feats)  # 将 dino_features 的最后一维映射到 clip_dim
            
        ex_feats /= ex_feats.norm(dim=-1, keepdim=True)
        logits = ex_feats.to(self.query_features.dtype) @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], I, J)
        '''
        #torch.save(self.query_features, 'text_feats.pt')
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], I, J)
        '''
        #######################
        prob = image_features[:, :1, :] @ self.query_features.T
        prob = (prob * 2).softmax(-1)
        w = prob / prob.mean(-1, keepdim=True)
        w = w.unsqueeze(-1)
        b, n_t, n_i, c = image_features.shape[0], self.query_features.shape[0], image_features.shape[1], image_features.shape[2]
        feats = image_features.reshape(b, n_i, 1, c) * self.query_features.reshape(1, 1, n_t, c)
        #print(w.shape,self.query_features.shape,image_features.shape,feats.shape)
        feats *= w
        redundant_feats = feats.mean(2, keepdim=True) # along cls dim
        feats = feats - redundant_feats
        # sum the element-wise multiplied features as cosine similarity
        sm = feats.sum(-1)[:, 1:, :]
        sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])

        # reshape
        side = int(sm.shape[1] ** 0.5) # square output
        sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)

        # interpolate
        sm = torch.nn.functional.interpolate(sm, img.shape[-2:], mode='bilinear')
        #return sm
        sm = sm.permute(0, 2, 3, 1)
        torch.save(sm, 'clip_surg.pt')
        #######################
        #######################
        target = image_features @ self.query_features.T
        sm1 = target[:, 1:, :]
        sm1 = (sm1 - sm1.min(1, keepdim=True)[0]) / (sm1.max(1, keepdim=True)[0] - sm1.min(1, keepdim=True)[0])

        # reshape
        side = int(sm1.shape[1] ** 0.5) # square output
        sm1 = sm1.reshape(sm1.shape[0], side, side, -1).permute(0, 3, 1, 2)

        # interpolate
        sm1 = torch.nn.functional.interpolate(sm1, img.shape[-2:], mode='bilinear')
        sm1 = sm1.permute(0, 2, 3, 1)
        torch.save(sm1, 'clip_no_surg.pt')
        #######################
        '''
        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')

        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)  # zero padding
                crop_seg_logit = self.forward_feature(crop_img).detach()

                torch.cuda.empty_cache()

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]
        
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])
        
        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices