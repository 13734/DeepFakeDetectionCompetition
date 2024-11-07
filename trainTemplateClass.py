from sympy.physics.units import momentum
from sympy.stats import moment

from dataset import Dataset_Loader
from torch.utils.data import DataLoader,ConcatDataset,Dataset
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
import time
import os
import shutil
import timm
from torch.cuda.amp import autocast,GradScaler
from trainlogger import TrainLogger
from tensorboardX import SummaryWriter
from  timm.utils import ModelEmaV3
import torch.utils.data as Data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 =True


scaler = GradScaler()
# 直接 MOE 模型路径
MODELS_PATH = "./models"

def accuracy(output, target, topk=(1,)):
    """
        计算topk的准确率
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        #output = torch.unsqueeze(output,0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        class_to = pred[0].cpu().numpy()

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # correct_k = correct[:k].contiguous.view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, class_to




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_data(file_path: str, train_flag=False):
        """

        :param file_path: str
                train_flag : bool
        :return: Dataset
        """
        file_names = os.listdir(file_path)
        list_dataset = []
        for i_str in file_names:
            list_dataset.append(Dataset_Loader(os.path.join(file_path,i_str),
                                               train_flag=train_flag))
        return ConcatDataset(list_dataset)


class TrainEngine:

    def __init__(self,train_model : nn.Module,my_model_name :str):

        self.my_model_name:str = my_model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

        # 定义基本参数

        batch_size =32
        self.total_epochs = 32

        lr_init = 1e-5
        lr_stepsize = 20
        weight_decay = 0.001
        # BF16
        self.USE_FP16 = 1

        # 加载数据集
        train_dir = './train_data'
        valid_dir = './val_data'
        total_num_workers = 6
        val_num_workers = 12

        #train_data = Dataset_Loader(train_dir_list, train_flag=True)
        #valid_data = Dataset_Loader(valid_dir_list, train_flag=False)
        train_data = get_data(train_dir,train_flag=True)
        valid_data = get_data(valid_dir,train_flag=False)

        train_data_size = len(train_data)
        print('训练集数量：%d' % train_data_size)
        valid_data_size = len(valid_data)
        print('验证集数量：%d' % valid_data_size)

        self.train_loader = DataLoader(dataset=train_data, num_workers=total_num_workers, pin_memory=True,
                                  batch_size=batch_size,
                                  shuffle=True)
        self.valid_loader = DataLoader(dataset=valid_data, num_workers=val_num_workers, pin_memory=True,
                                  batch_size=batch_size)

        # 定义损失函数和优化器等
        self.model : nn.Module= train_model
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(device=self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr_init, weight_decay=weight_decay)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=3e-4,momentum=0.9, weight_decay=5e-4)

        #self.optimizer = optim.RAdam(self.model.parameters(), lr=lr_init, weight_decay=weight_decay)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_stepsize, gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,start_factor=0.001,total_iters=1000)

        # 写入日志
        self.my_train_logger = TrainLogger()
        self.writer = SummaryWriter('runs/' + my_model_name)
        self.my_train_logger.trainStart(my_model_name)

        #准备路径
        self.TRADITION_MODELS_PATH = "./outputs"

        if not os.path.exists(self.TRADITION_MODELS_PATH):
            os.mkdir(self.TRADITION_MODELS_PATH)
        self.step_model_path = "./step_output"
        if not os.path.exists(self.step_model_path):
            os.mkdir(self.step_model_path)

        # 按步数验证准备
        self.step_train: int = 0
        self.save_steps: int = 6000

        self.best_prec_step = 0
        self.best_prec_step_ema = 0

        # EMA
        USE_EMA = True
        self.model_ema = None
        if  USE_EMA:
            self.model_ema = ModelEmaV3( self.model,decay=0.9995,use_warmup=True,device=self.device)




    def save_model(self,is_best,isEma =  False):
        if isEma:
            state_dict = self.model_ema.module.state_dict()
            filename = "_step_" + self.my_model_name + "_ema.pth.tar"
        else:
            state_dict = self.model.state_dict()
            filename = "_step_" + self.my_model_name + ".pth.tar"
        state = {
            'step': self.step_train + 1,
            'arch': self.my_model_name,
            'state_dict': state_dict,
            'best_prec': self.best_prec_step,
            'optimizer': self.optimizer.state_dict(),
        }

        step = str(state["step"] - 1)
        filepath = os.path.join(self.step_model_path, step + filename)

        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(self.step_model_path, 'model_best_' + filename))

    def step_step(self):

        if self.step_train > 0 and self.step_train % self.save_steps == 0:
            is_best = False
            val_predict,_ =  self.validate(self.step_train,"VAL_Step")
            if val_predict > self.best_prec_step:
                is_best = True
                self.best_prec_step = val_predict
            self.save_model(is_best, False)

            is_best_ema = False
            val_predict_ema, _ = self.validate(self.step_train, "VAL_Step_ema",this_model= self.model_ema)
            if val_predict_ema > self.best_prec_step_ema:
                is_best_ema = True
                self.best_prec_step_ema = val_predict_ema
            self.save_model(is_best_ema,True)



    def save_checkpoint(self,state, is_best, filename='checkpoint.pth.tar'):
        """

        """
        epoch = str(state["epoch"]-1)
        filepath = os.path.join(self.TRADITION_MODELS_PATH,epoch+"_"+filename)

        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(self.TRADITION_MODELS_PATH,'model_best_' + filename))
    #

    def train_per_epoch(self, epoch:int):


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (input, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(self.device)

            #arget = target.float()
            target = target.to(self.device)
            self.optimizer.zero_grad()
            # compute output
            if self.USE_FP16:
                with autocast():
                    output = self.model(input)
                    loss = self.criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                output = self.model(input)
                loss = self.criterion(output, target)
                # compute gradient and do SGD step
                loss.backward()
                self.optimizer.step()

            # measure accuracy and record loss
            [prec1, prec5], class_to = accuracy(output, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            if self.model_ema:
                self.model_ema.update(self.model)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))



            self.writer.add_scalar("loss/train/per_step",losses.val,global_step=self.step_train)
            self.writer.add_scalar("accuracu/train/per_step",top1.val, global_step = self.step_train)
            self.step_train += 1
            self.step_step()
            # update per  step
            self.scheduler.step()

        self.writer.add_scalar('loss/train_loss', losses.val, global_step=epoch)

    def validate(self, epoch, phase="VAL",this_model:nn.Module = None):

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        if not this_model:
            this_model = self.model

        # switch to evaluate mode
        this_model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.valid_loader):
                input = input.to(self.device)
                target = target.to(self.device)
                # compute output
                if self.USE_FP16:
                    with autocast():
                        output = this_model(input)
                        loss = self.criterion(output, target)
                else:
                    output = this_model(input)
                    loss = self.criterion(output, target)

                # measure accuracy and record loss
                [prec1, prec5], class_to = accuracy(output, target, topk=(1, 1))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    print('Test-{0}: [{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        phase, i, len(self.valid_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1, top5=top5))

            print(' * {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(phase, top1=top1, top5=top5))
        self.writer.add_scalar(phase+'loss/valid_loss', losses.val, global_step=epoch)
        self.my_train_logger.writeTrainInfo({
            "mode"  : phase,
            "train" : "",
            "step": self.step_train,
            "epoch": epoch,
            "loss" : "{:.3f}".format(losses.avg),
            "accurate": "{:.3f}".format(top1.avg)
        })
        this_model.train()
        return top1.avg, top5.avg

    #def train_main(self,train_model :nn.Module,my_model_name:str):
    def train_main(self):

        best_prec1 = 0
        best_prec1_ema = 0
        for epoch in range(self.total_epochs):
            # scheduler.step()
            self.train_per_epoch(epoch)

            valid_prec1, valid_prec5 = self.validate( epoch,phase="VAL")

            valid_prec1_ema,_ =self.validate(epoch,phase="VAL_EMA",this_model = self.model_ema)


            if self.my_model_name == "mff_moe":
                for i in range(len(self.model.ema_list)):
                    self.model.ema_list[i].update(self.model.experts[i].parameters())
                print("EMA_VAL")
                self.model.not_use_ema = False
                valid_prec1, valid_prec5 = self.validate(epoch, phase="EMA_VAL")
                self.model.not_use_ema = True

            is_best = valid_prec1 > best_prec1
            best_prec1 = max(valid_prec1, best_prec1)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.my_model_name,
                'state_dict': self.model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': self.optimizer.state_dict(),
            }, is_best,
                filename='checkpoint_{}.pth.tar'.format(self.my_model_name))


            is_best_ema = valid_prec1_ema > best_prec1_ema
            best_prec1_ema = max(valid_prec1_ema, best_prec1_ema)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.my_model_name,
                'state_dict': self.model_ema.module.state_dict(),
                'best_prec1': best_prec1_ema,
                'optimizer': self.optimizer.state_dict(),
            }, is_best,
                filename='checkpoint_ema_{}.pth.tar'.format(self.my_model_name))

        self.writer.close()


if __name__ == '__main__':




    my_mode = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=True,num_classes=2)

    train_model = TrainEngine(my_mode, "convnextv2_tiny-base")
    train_model.train_main()

    
