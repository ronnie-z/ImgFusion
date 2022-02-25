import torch
from torch.utils import data
from torchvision import transforms
import  numpy as np
from net import *
from dataset import *
from torchsummary import summary

class ToTensor(object):
    def __call__(self, img):
        img = np.transpose(img.astype(np.float32), (2, 0, 1))

        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img / 255.0
        tensor = torch.from_numpy(img).float()
        return tensor

train_trainsform = transforms.Compose([
        ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5]),
        # transforms.RandomCrop(size=64, pad_if_needed=True),
    ])

val_dataset = DataSet('D:\code\imagefusion-rfn-nest\images/21_pairs_tno/vis', train_trainsform)

val_loader = data.DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers = 2, drop_last = True)



def save_img(img, count):
    img_fusion = img.float()

    img_fusion = img_fusion.cpu().data.numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    cv2.imwrite('./result_first_train_e0/img_fusion_%d.jpg' % count ,img_fusion)
    # if img_fusion.shape[2] == 1:  # 如果是灰度图
    #     img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])

def load_pretrain(model, ckpt_path):

    if not torch.cuda.is_available():
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))   #模型加载到GPU上


    model.load_state_dict(ckpt_dict, strict=False)   #加载预训练模型到model
    return model


def main(ckpt_path):
    count = 0
    netG = Generator()
    netG = load_pretrain(netG, ckpt_path)
    with torch.no_grad():
        netG.eval()
        netG.cuda()
        # summary(netG, [(1, 128, 128), (1, 128, 128)])
        for data in val_loader:
            vis_img = data[0].cuda()
            ir_img = data[1].cuda()
            vis_list = netG.encoder(vis_img)
            ir_list = netG.encoder(ir_img)  # [g1,g2,g3,x3]
            fusion_img = netG.decoder_eval(vis_list, ir_list)

            for img in fusion_img:
                count += 1
                save_img(img, count)



if __name__ == '__main__':
    ckpt_path = 'model_output/checkpoint_e0_Gen.pth'
    main(ckpt_path)
    # ckpt_dict = torch.load(ckpt_path)
    # for k,v in ckpt_dict.items():
    #     print(k,'\t', v.shape)