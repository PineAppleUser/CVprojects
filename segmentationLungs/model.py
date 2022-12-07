import torch
import torch.nn as nn
#import torchvision

class unet(nn.Module):

  def __init__(self):
    super().__init__()

    #Encoder layers
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2)
    self.conv11 = nn.Conv2d(1, 64, 3, 1, 1, bias = False)
    self.btch_nrm64 = nn.BatchNorm2d(64)
    self.conv12 = nn.Conv2d(64, 64, 3, 1, 1, bias = False)
    self.conv21 = nn.Conv2d(64, 128, 3, 1, 1, bias = False)
    self.btch_nrm128 = nn.BatchNorm2d(128)
    self.conv22 = nn.Conv2d(128, 128, 3, 1, 1, bias = False)
    self.conv31 = nn.Conv2d(128, 256, 3, 1, 1, bias = False)
    self.btch_nrm256 = nn.BatchNorm2d(256)
    self.conv32 = nn.Conv2d(256, 256, 3, 1, 1, bias = False)
    self.conv41 = nn.Conv2d(256, 512, 3, 1, 1, bias = False)
    self.btch_nrm512 = nn.BatchNorm2d(512)
    self.conv42 = nn.Conv2d(512, 512, 3, 1, 1, bias = False)
    self.conv51 = nn.Conv2d(512, 1024, 3, 1, 1, bias = False)
    self.btch_nrm1024 = nn.BatchNorm2d(1024)
    self.conv52 = nn.Conv2d(1024, 1024, 3, 1, 1, bias = False)


    #Decoder layers
    self.convTrans1 = nn.ConvTranspose2d(1024, 512, 2, 2)
    self.dconv11 = nn.Conv2d(1024, 512, 3, 1, 1, bias = False)
    self.dconv12 = nn.Conv2d(512, 512, 3, 1, 1, bias = False)

    self.convTrans2 = nn.ConvTranspose2d(512, 256, 2, 2)
    self.dconv21 = nn.Conv2d(512, 256, 3, 1, 1, bias = False)
    self.dconv22 = nn.Conv2d(256, 256, 3, 1, 1, bias = False)

    self.convTrans3 = nn.ConvTranspose2d(256, 128, 2, 2)
    self.dconv31 = nn.Conv2d(256, 128, 3, 1, 1, bias = False)
    self.dconv32 = nn.Conv2d(128, 128, 3, 1, 1, bias = False)

    self.convTrans4 = nn.ConvTranspose2d(128, 64, 2, 2)
    self.dconv41 = nn.Conv2d(128, 64, 3, 1, 1, bias = False)
    self.dconv42 = nn.Conv2d(64, 64, 3, 1, 1, bias = False)

    #last layer
    self.conv_last = nn.Conv2d(64, 1, 1)

  def forward(self, input):
    #enc
    x1 = self.relu( self.btch_nrm64(self.conv12( self.relu( self.btch_nrm64( self.conv11( input ))))))
    p1 = self.pool(x1)

    x2 = self.relu( self.btch_nrm128(self.conv22( self.relu( self.btch_nrm128( self.conv21( p1 ))))))
    p2 = self.pool(x2)

    x3 = self.relu( self.btch_nrm256(self.conv32( self.relu( self.btch_nrm256( self.conv31( p2 ))))))
    p3 = self.pool(x3)

    x4 = self.relu( self.btch_nrm512(self.conv42( self.relu( self.btch_nrm512(self.conv41( p3 ))))))
    p4 = self.pool(x4)

    #trns
    x5 = self.relu( self.btch_nrm1024(self.conv52( self.relu( self.btch_nrm1024(self.conv51( p4 ))))))

    #dec

    #cropping commented out since decided to use padding instead, which proved to be not worse at all researching usage on some serious competitions
    #and providing are more convinients results to deal with

    up_conv1 = self.convTrans1(x5)
    #x4_cropped = self.crop(x4, up_conv1)
    #y1 = torch.cat([up_conv1, x4_cropped], dim=1)
    y1 = torch.cat([up_conv1, x4], dim=1)
    y1 = self.relu( self.btch_nrm512(self.dconv12 ( self.relu ( self.btch_nrm512(self.dconv11( y1 ))))))

    up_conv2 = self.convTrans2(y1)
    #x3_cropped = self.crop(x3, up_conv2)
    #y2 = torch.cat([up_conv2, x3_cropped], dim=1)
    y2 = torch.cat([up_conv2, x3], dim=1)
    y2 = self.relu( self.btch_nrm256(self.dconv22 ( self.relu ( self.btch_nrm256(self.dconv21( y2 ))))))

    up_conv3 = self.convTrans3(y2)
    #x2_cropped = self.crop(x2, up_conv3)
    #y3 = torch.cat([up_conv3, x2_cropped], dim=1)
    y3 = torch.cat([up_conv3, x2], dim=1)
    y3 = self.relu( self.btch_nrm128(self.dconv32 ( self.relu ( self.btch_nrm128(self.dconv31( y3 ))))))

    up_conv4 = self.convTrans4(y3)
    #x1_cropped = self.crop(x1, up_conv4)
    #y4 = torch.cat([up_conv4, x1_cropped], dim=1)
    y4 = torch.cat([up_conv4, x1], dim=1)
    y4 = self.relu( self.btch_nrm64(self.dconv42 ( self.relu ( self.btch_nrm64(self.dconv41( y4 ))))))

    #fin
    output =  self.conv_last(y4)

    return output
'''
  def crop(self, enc_ftrs, x):
    _, _, H, W = x.shape
    enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
    return enc_ftrs
'''