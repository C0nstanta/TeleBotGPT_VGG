from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import requests
import copy
import os
import io


layers_dict = {'0': "conv1_1(3, 64)", '1': "conv1_2(64, 64)", '2': "conv2_1(64,128)", '3': "conv2_2(128,128)",
               '4': "conv3_1(128,256)", '5': "conv3_2(256,256)", '6': "conv3_3(256,256)", '7': "conv3_4(256,256)",
               '8': "conv4_1(256,512)", '9': "conv4_2(512,512)", '10': "conv4_3(512,512)", '11': "conv4_4(512,512)",
               '12': "conv5_1(512,512)", '13': "conv5_2(512,512)",'14':"conv5_3(512,512)", '15': "conv5_4(512,512)"}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    g = torch.mm(features, features.t())
    return g.div(a * b * c * d)


class ImagePreprocessing:
    def __init__(self, imsize=512):
        self.loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])


    def im_crop_center(self, img):
        img_width, img_height = img.size
        w = h = min(img_width, img_height)
        left, right = (img_width - w) / 2, (img_width + w) / 2
        top, bottom = (img_height - h) / 2, (img_height + h) / 2
        left, top = round(max(0, left)), round(max(0, top))
        right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
        return img.crop((left, top, right, bottom))


    def image_loader(self, file_path):
        r = requests.get(file_path, stream=True)
        if r.status_code == 200:
            image = Image.open(io.BytesIO(r.content))
            image = self.im_crop_center(image, )

            print(type(image))
            image = self.loader(image).unsqueeze(0)
            return image.to(device, dtype=torch.float)
        else:
            return None


    def im_detorch(self, img):
        img = img.clone()
        img = img.squeeze(0)
        unloader = transforms.ToPILImage()
        img = unloader(img)
        return img


    def im_to_bytearray(self, img):
        rgb_im = img.convert("RGB")

        img_byte_arr = io.BytesIO()
        rgb_im.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr



class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()


    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_features).detach()


    def forward(self, input):
        g = gram_matrix(input)
        self.loss = F.mse_loss(g, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).clone().detach().view(-1, 1, 1)
        self.std = torch.tensor(std).clone().detach().view(-1, 1, 1)


    def forward(self, img):
        return (img - self.mean) / self.std


class VggModel:
    base_cont_layer = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    base_style_layer = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def __init__(self, device="cpu"):
        my_dir = os.getcwd()
        os.environ['TORCH_HOME'] = my_dir + "/model_vgg19/"

        self.cnn = models.vgg19(pretrained=True).features.to(device).eval()
        self.cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


    def get_input_optimizer(self, input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer


    def change_st_ct_layer(self):
        pass


    def get_style_model_and_losses(self, st_img, ct_img, style_layers, content_layers):
        cont_layer = ["conv_" + str(idx + 1) for idx, val in enumerate(content_layers) if val == 1]# VggModel.base_cont_layer
        style_layer = ["conv_" + str(idx + 1) for idx, val in enumerate(style_layers) if val == 1]# VggModel.base_style_layer

        cnn = copy.deepcopy(self.cnn)
        norm = Normalization(self.cnn_norm_mean, self.cnn_norm_std).to(device)
        ct_losses = []
        st_losses = []

        model = nn.Sequential(norm)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer')
                break
            model.add_module(name, layer)

            if name in cont_layer:
                target = model(ct_img).detach()
                ct_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), ct_loss)
                ct_losses.append(ct_loss)

            if name in style_layer:
                target_st = model(st_img).detach()
                st_loss = StyleLoss(target_st)
                model.add_module("style_loss_{}".format(i), st_loss)
                st_losses.append(st_loss)
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]
        return model, st_losses, ct_losses


    def run_style_transfer(self, content_img, style_img, input_img, num_steps=1000,
                           style_weight=1000000, content_weight=1, show_every=200, style_layers=base_style_layer,
                           content_layers=base_cont_layer):
        """Run the style transfer."""
        print('Building the style transfer model')
        model, style_losses, content_losses = self.get_style_model_and_losses(style_img, content_img,
                                                                              style_layers, content_layers)
        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % show_every == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                    print(type(style_score))
                return style_score + content_score

            optimizer.step(closure)
        # a last correction...
        input_img.data.clamp_(0, 1)
        return input_img


# def get_output(style_image=None, content_image=None, num_steps=1):
#     image_process = ImagePreprocessing()
#
#     style_img = image_process.image_loader(
#         file_path="https://api.telegram.org/file/bot1900850036:AAHinBkKM69bwZRVYdHqJMHQS3ePx2ZLSwg/photos/file_8.jpg")
#     content_img = image_process.image_loader(
#         file_path="https://storage.ws.pho.to/s2/4fc46cab6349ea8ca7d8f932ba2b15a9445a146e_m.jpeg")
#     input_img = content_img
#
#     model = VggModel()
#     output = model.run_style_transfer(content_img=content_img, style_img=style_img, input_img=input_img, num_steps=num_steps)
#
#     return output


# if __name__ == "__main__":
    # image_process = ImagePreprocessing()
    #
    # style_img = image_process.image_loader(file_path="https://api.telegram.org/file/bot1900850036:AAHinBkKM69bwZRVYdHqJMHQS3ePx2ZLSwg/photos/file_8.jpg")
    # content_img = image_process.image_loader(file_path="https://storage.ws.pho.to/s2/4fc46cab6349ea8ca7d8f932ba2b15a9445a146e_m.jpeg")
    # input_img = content_img
    #
    # model = VggModel()
    #
    # output = model.run_style_transfer(content_img, style_img, input_img)
    # output = get_output()