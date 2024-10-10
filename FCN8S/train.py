import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
import pandas as pd
from dataset import CamvidDataset
from evalution_segmentaion import eval_semantic_segmentation
from FCN import FCN
import cfg

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

Cam_train = CamvidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Cam_val = CamvidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Cam_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
val_data = DataLoader(Cam_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

fcn = FCN(2)
fcn = fcn.to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(fcn.parameters(), lr=1e-4)


def train(model):
    best = [0]
    net = model.train()
    # 训练轮次
    result_data = pd.DataFrame(columns=['train_acc', 'train_loss', 'train_miou', 'train_class_acc', 'val_acc', 'val_loss', 'val_miou', 'val_class_acc'])

    for epoch in range(cfg.EPOCH_NUMBER):
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.1

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = Variable(sample['img'].to(device))   
            img_label = Variable(sample['label'].to(device))    
            # 训练
            out = net(img_data)     
            out = F.log_softmax(out, dim=1)

            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()    
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()   
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            train_class_acc += eval_metrix['class_accuracy']

            # print('|batch[{}/{}]--|batch_loss {: .8f}|--|batch_acc {: .8f}--|batch_miou {: .8f}'.format(i + 1, len(train_data), loss.item(), eval_metrix['mean_class_accuracy'], eval_metrix['miou']))

        metric_description = '|Train Acc|: {:.5f}--|Train loss|: {:.5f}--|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}'.format(
            train_acc / len(train_data),
            train_loss / len(train_data),
            train_miou / len(train_data),
            train_class_acc / len(train_data),
        )

        print(metric_description)
        if max(best) <= train_miou / len(train_data):
            best.append(train_miou / len(train_data))
            t.save(net.state_dict(), './FCN_train/model{}.pth'.format(epoch))

        val_data = evaluate(net)
        result_data.loc[epoch] = train_acc / len(train_data), train_loss / len(train_data), train_miou / len(train_data), train_class_acc / len(train_data), val_data[1], val_data[0], val_data[2], val_data[3]
    result_data.to_csv('./result_data.csv')

def evaluate(model):
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    eval_class_acc = 0

    prec_time = datetime.now()
    for j, sample in enumerate(val_data):
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].long().to(device))

        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)
        eval_loss = loss.item() + eval_loss
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrics = eval_semantic_segmentation(pre_label, true_label)
        eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
        eval_miou = eval_metrics['miou'] + eval_miou
        eval_class_acc = eval_metrics['class_accuracy'] + eval_class_acc

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)

    val_str = ('|Valid Loss|: {:.5f} |Valid Acc|: {:.5f} |Valid Mean IU|: {:.5f} \n|Valid Class Acc|:{:}'.format(
        eval_loss / len(val_data),
        eval_acc / len(val_data),
        eval_miou / len(val_data),
        eval_class_acc / len(val_data)))
    print(val_str)
    print(time_str)
    return [eval_loss / len(val_data), eval_acc / len(val_data), eval_miou / len(val_data), eval_class_acc / len(val_data)]


if __name__ == "__main__":

    train(fcn)

