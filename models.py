import torch 
import torch .nn as nn 


class Dicriminator(nn.Module):
    def __init__(self, channel,num_label,image_size,filters=[64, 128 ,256 ,512]):
        super().__init__()   
        self.image_size = image_size
        self.embed = nn.Embedding(num_label,image_size*image_size)

        self.disc=nn.Sequential(
            self.block((channel*2)+1, filters[0] , 4 , 2 , 1 ,False),
            self.block(filters[0] , filters[1] , 4 , 2 , 1), 
            self.block(filters[1] , filters[2] , 4 , 2 , 1),
            self.block(filters[2], filters[3] , 4 , 1 , 1),
            nn.Conv2d(filters[3] , 1 , 4 ,1 ,1 , padding_mode='reflect')
        )



    def block(self , in_ch , out_ch , kernel , stride , padding , batch=True):
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False ,padding_mode="reflect")]
        if batch:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self,x,y,label):
        image = torch.cat([x,y], dim=1)
        em = self.embed(label).view(label.shape[0] , 1 , self.image_size , self.image_size)
        image = torch.cat([image,em] ,dim=1)
        return self.disc(image)
    

class Genrator(nn.Module):
    def __init__(self, channel,num_label ,image_size, filters=64):
        super().__init__()
        self.image_size = image_size
        self.embed = nn.Embedding(num_label,image_size*image_size)
        self.init_down = nn.Sequential(
            nn.Conv2d(channel+1 , filters , 4 ,2 ,1 ,padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        self.down1=self.block(filters , filters*2 , 4 ,2 ,1)
        self.down2=self.block(filters*2 , filters*4 , 4 ,2 ,1)
        self.down3=self.block(filters*4 , filters*8 , 4 ,2 ,1)
        self.down4=self.block(filters*8 , filters*8 , 4 ,2 ,1)
        self.down5=self.block(filters*8 , filters*8 , 4 ,2 ,1)
        self.down6=self.block(filters*8 , filters*8 , 4 ,2 ,1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(filters*8,filters*8,4,2,1 , bias=False ,padding_mode="reflect"),
            nn.ReLU()
        )
        self.up1=self.block(filters*8 , filters*8 , 4 ,2 ,1 , "relu" , False , True)
        self.up2=self.block(filters*8*2 , filters*8 , 4 ,2 ,1 , "relu" , False , True)
        self.up3=self.block(filters*8*2 , filters*8 , 4 ,2 ,1 , "relu" , False , True)
        self.up4=self.block(filters*8*2 , filters*8 , 4 ,2 ,1 , "relu" , False , False)
        self.up5=self.block(filters*8*2 , filters*4 , 4 ,2 ,1 , "relu" , False , False)
        self.up6=self.block(filters*4*2 , filters*2 , 4 ,2 ,1 , "relu" , False , False)
        self.up7=self.block(filters*2*2 , filters , 4 ,2 ,1 , "relu" , False , False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(filters*2,channel,4,2,1 , bias=False),
            nn.Tanh()
        )





    def block(self, in_ch, out_ch, kernel, stride, padding, act="lea", down=True, drop=False):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False, padding_mode="reflect")
            if down
            else
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        ]

        if drop:
            layers.append(nn.Dropout(0.5))  # Adjust the dropout rate as needed

        x = nn.Sequential(*layers)

        return x
    
    
    def forward(self, x,label):
        em = self.embed(label).view(label.shape[0] , 1 , self.image_size , self.image_size)
        x = torch.cat([x,em] ,dim=1)
        d1 = self.init_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final(torch.cat([up7, d1], 1))
    


def init_weights(m):
    if isinstance(m,(nn.Conv2d , nn.ConvTranspose2d ,nn.BatchNorm2d )):
        nn.init.normal_(m.weight , mean=0 ,std=0.02)



