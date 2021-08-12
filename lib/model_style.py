from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

import torch
import torch.optim as optim


import requests
import os
import io


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
            print()
            image = self.loader(image).unsqueeze(0)
            print(image.shape)
            return image.to(device, dtype=torch.float)
        else:
            return False


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


class Vgg:
    def __init__(self):
        my_dir = os.getcwd()
        os.environ['TORCH_HOME'] = my_dir + "/model_vgg19/"
        self.model = models.vgg19(pretrained=True).features

    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad_(False)


    def get_features(self, image,  layers=None):
        """ Run an image forward through a model and get the features for
            a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
        """

        ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
        ## Need the layers for the content and style representations of an image
        if layers is None:
            layers = {'0': 'conv1_1',
                      '5': 'conv2_1',
                      '10': 'conv3_1',
                      '19': 'conv4_1',
                      '21': 'conv4_2',  ## content representation
                      '28': 'conv5_1'}

        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features


    def gram_matrix(self, tensor):
        """ Calculate the Gram Matrix of a given tensor
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """

        # get the batch_size, depth, height, and width of the Tensor
        _, d, h, w = tensor.size()

        # reshape so we're multiplying the features for each channel
        tensor = tensor.view(d, h * w)

        # calculate the gram matrix
        gram = torch.mm(tensor, tensor.t())
        return gram


    def run(self, content=None, style_features=None, content_features=None, steps=5, show_every=1):
        # create a third "target" image and prep it for change
        # it is a good idea to start of with the target as a copy of our *content* image
        # then iteratively change its style
        target = content.clone().requires_grad_(True).to(device)

        # calculate the gram matrices for each layer of our style representation
        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}
        style_weights = {'conv1_1': 1.,
                         'conv2_1': 0.75,
                         'conv3_1': 0.2,
                         'conv4_1': 0.2,
                         'conv5_1': 0.2}

        content_weight = 1  # alpha
        style_weight = 1e9  # beta

        optimizer = optim.Adam([target], lr=0.003)

        # for displaying the target image, intermittently
        show_every = show_every

        steps = steps  # decide how many iterations to update your image (5000)

        for ii in range(1, steps + 1):

            # get the features from your target image
            target_features = self.get_features(target)

            # the content loss
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

            # the style loss
            # initialize the style loss to 0
            style_loss = 0
            # then add to it for each layer's gram matrix loss
            for layer in style_weights:
                # get the "target" style representation for the layer
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                # get the "style" style representation
                style_gram = style_grams[layer]
                # the style loss for one layer, weighted appropriately
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)

            # calculate the *total* loss
            total_loss = content_weight * content_loss + style_weight * style_loss

            # update your target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # display intermediate images and print the loss
            if ii % show_every == 0:
                print('Total loss: ', total_loss.item())
                print("target: ", target.shape)

        return target


# if __name__ == "__main__":
#     image_process = ImagePreprocessing()
#     style = image_process.image_loader(file_path="https://api.telegram.org/file/bot1900850036:AAHinBkKM69bwZRVYdHqJMHQS3ePx2ZLSwg/photos/file_8.jpg")
#     content = image_process.image_loader(file_path="https://storage.ws.pho.to/s2/4fc46cab6349ea8ca7d8f932ba2b15a9445a146e_m.jpeg")
#
#     model = Vgg()
#     model.freeze_layers()
#
#     # get content and style features only once before training
#     content_features = model.get_features(content)
#     style_features = model.get_features(style)
#
#
#     # create a third "target" image and prep it for change
#     # it is a good idea to start of with the target as a copy of our *content* image
#     # then iteratively change its style
#     # target = content.clone().requires_grad_(True).to(device)
#     img = model.run(content_features=content_features, style_features=style_features, steps=3, show_every=1)
#     print(img.shape)
'''
vgg._modules.items()
odict_items([
!!!style     ('0', Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('1', ReLU(inplace=True)),
             ('2', Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('3', ReLU(inplace=True)),
             ('4', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
!!!style     ('5', Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('6', ReLU(inplace=True)),
             ('7', Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('8', ReLU(inplace=True)),
             ('9', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
!!!style     ('10', Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('11', ReLU(inplace=True)),
             ('12', Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('13', ReLU(inplace=True)),
             ('14', Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('15', ReLU(inplace=True)),
             ('16', Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('17', ReLU(inplace=True)),
             ('18', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
!!!style     ('19', Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('20', ReLU(inplace=True)),
!!!content   ('21', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('22', ReLU(inplace=True)),
             ('23', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('24', ReLU(inplace=True)),
             ('25', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('26', ReLU(inplace=True)),
             ('27', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
!!!style     ('28', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('29', ReLU(inplace=True)),
             ('30', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('31', ReLU(inplace=True)),
             ('32', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('33', ReLU(inplace=True)),
             ('34', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
             ('35', ReLU(inplace=True)),
             ('36', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))])
16 layers Conv2d
'''