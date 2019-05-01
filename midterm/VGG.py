import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,num_classes=10,in_channels=3, init_weights=True):
        super(VGG,self).__init__()
        pc = 0.0
        pl = 0.5
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(p=pc),
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1,padding=0))
            nn.MaxPool2d(kernel_size=2,stride=2))
	    # nn.AvgPool2d(kernel_size=2,stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(p=pc),
            nn.MaxPool2d(kernel_size=2,stride=2))
            #nn.AvgPool2d(kernel_size=2,stride=2))
            #nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1,padding=1))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(p=pc),
            #nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1,padding=0))
            nn.MaxPool2d(kernel_size=2,stride=2))
            #nn.AvgPool2d(kernel_size=2,stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(p=pc),
            #nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1,padding=0))
            nn.MaxPool2d(kernel_size=2,stride=2))
            #nn.AvgPool2d(kernel_size=2,stride=2))

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(p=pc),
            #nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1,padding=0))
            nn.MaxPool2d(kernel_size=2,stride=2))
            #nn.AvgPool2d(kernel_size=2,stride=2))

        #self.avgpool = nn.Conv2d((in_channels=512, out_channels=512, kernel_size=7, stride=7, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
           # nn.Dropout(p=pl),
            nn.Linear(512 * 7 * 7,4096),
            nn.ReLU(True),
            nn.Dropout(p=pl),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(p=pl),
           # nn.Linear(4096,256),
           # nn.ReLU(True),
           # nn.Dropout(p=pl),
            nn.Linear(4096,num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
                  if m.bias is not None:
                      nn.init.constant_(m.bias, 0)
              elif isinstance(m, nn.BatchNorm2d):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
              elif isinstance(m, nn.Linear):
                  nn.init.normal_(m.weight, 0, 0.002)
                  nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
	
