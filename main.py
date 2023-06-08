import torch
import torchvision
from torchvision import transforms, models
from PIL import  Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import copy
import pylab #为了输出图片
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #解决将图片转为RGB模式后的危险报错


img_path = "mai.jpg"
save_path = 'mai.jpg'
img = Image.open(img_path)
img = img.convert("RGB") #将内容图片转为RGB模式降低维度
img.save(save_path)
img_path = "style4.jpg"
save_path = 'style4.jpg'
img = Image.open(img_path)
img = img.convert("RGB") #将风格图片转为RGB模式降低维度
img.save(save_path)


transform = transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()]) #设置输出图片大小
def loadimg(path=None): #设置读取图片函数
    img = Image.open(path)
    img = transform(img)
    img = img.unsqueeze(0)
    return img

ToPIL = torchvision.transforms.ToPILImage() #转为torch

def img_show(img, title=None): #设置输出图片
    img = img.clone().cpu()
    img = img.view(3, 224, 224)
    img = ToPIL(img)

    if title is not None:
        plt.title(title)
    plt.imshow(img)

content_img = loadimg("mai.jpg") #读取已经保存在项目文件下的图片
content_img = Variable(content_img).cuda()
style_img = loadimg("style4.jpg")
style_img = Variable(style_img).cuda()

print(content_img.size()) #输出图片tensor中的张量维度

plt.figure() #输出原始图片
img_show(content_img.data, title = "Content_Image")
plt.figure()
img_show(style_img.data, title = "Style_Image")


class Content_loss(torch.nn.Module): #定义内容Loss,风格Loss和Gram矩阵
    def __init__(self, weight, target):
        super(Content_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input):
        self.loss = self.loss_fn(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss

class gram_matrix(torch.nn.Module):
    def forward(self, input):
        a,b,c,d = input.size()
        feature = input.view(a*b, c*d)
        gram = torch.mm(feature, feature.t())
        return gram.div(a*b*c*d)


class Style_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Style_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()
        self.gram = gram_matrix()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.loss_fn(self.G, self.target)
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss

use_gpu = torch.cuda.is_available() #定义需要优化的参数
cnn = models.vgg19(pretrained=True).features
if use_gpu:
    cnn = cnn.cuda()

print(cnn)


content_layer = ["Conv_5", "Conv_6"]

style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4", "Conv_5"]

content_losses = []
style_losses = []

conten_weight = 1
style_weight = 1000

new_model = torch.nn.Sequential()

model = copy.deepcopy(cnn)

gram = gram_matrix()

if use_gpu:
    new_model = new_model.cuda()
    gram = gram.cuda()

index = 1
for layer in list(model):
    if isinstance(layer, torch.nn.Conv2d):
        name = "Conv_" + str(index)
        new_model.add_module(name, layer)
        if name in content_layer:
            target = new_model(content_img).clone()
            content_loss = Content_loss(conten_weight, target)
            new_model.add_module("content_loss_" + str(index), content_loss)
            content_losses.append(content_loss)

        if name in style_layer:
            target = new_model(style_img).clone()
            target = gram(target)
            style_loss = Style_loss(style_weight, target)
            new_model.add_module("style_loss_" + str(index), style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, torch.nn.ReLU):
        name = "Relu_" + str(index)
        new_model.add_module(name, layer)
        index = index + 1

    if isinstance(layer, torch.nn.MaxPool2d):
        name = "MaxPool_" + str(index)
        new_model.add_module(name, layer)

print(new_model)

input_img = content_img.clone()

parameter = torch.nn.Parameter(input_img.data)
optimizer = torch.optim.LBFGS([parameter])


n_epoch = 100 #设置训练次数

run = [0]
while run[0] <= n_epoch:

    def closure():
        optimizer.zero_grad()
        style_score = 0
        content_score = 0
        parameter.data.clamp_(0, 1)
        new_model(parameter)
        for sl in style_losses:
            style_score += sl.backward()

        for cl in content_losses:
            content_score += cl.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            print('{} Style Loss : {:4f} Content Loss: {:4f}'.format(run,style_score.data, content_score.data)) #输出训练过程中的风格损失，每50次输出一次

        return style_score + content_score


    optimizer.step(closure)

parameter.data.clamp_(0,1) #训练迁移结束，输出画布和结果图
plt.figure()
img_show(parameter.data, title="Output Image")
pylab.show()