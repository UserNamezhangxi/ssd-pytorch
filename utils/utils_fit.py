import math
import os.path
import torch

from utils.utils import get_lr


def fit_one_epoch(model, loss_fn, train_data, valid_data,  epoch_train_step, epoch_valid_step, optimizer, epoch, total_epoch, writer, device, save_period, save_dir):
    train_loss = 0
    val_loss = 0
    los_file = open('./logs.txt', 'a', encoding='utf-8')
    model.train()
    for iteration, batch in enumerate(train_data):
        image, boxes = batch

        image = image.to(device)
        boxes = boxes.to(device)
        # 前向传播
        outputs = model(image)
        # 清零梯度
        optimizer.zero_grad()
        # 计算损失  loss = ssd_loss.forward(targets, outpus)
        loss = loss_fn(boxes, outputs)
        # 反向传播
        loss.backward()
        # 更新优化器
        optimizer.step()

        train_loss += loss.item()
        print("=%d/%d= total loss:%.3f, lr:%f, iteration:%3d"%(epoch + 1, total_epoch, train_loss/(iteration+1), get_lr(optimizer), iteration))
        if math.isnan(train_loss):
            los_file.write("=%d/%d= total loss:%.3f, lr:%f, iteration:%3d"%(epoch + 1, total_epoch, train_loss/(iteration+1), get_lr(optimizer), iteration))
            los_file.write("\r\n")

    writer.add_scalar("train_loss", train_loss / epoch_train_step , epoch)
    print('Finish Training')
    print('Start Validation')

    # 验证集验证模型
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(valid_data):
            image, boxes = batch
            image = image.to(device)
            boxes = boxes.to(device)
            # 前向传播
            outputs = model(image)
            # 梯度清零
            optimizer.zero_grad()
            # 计算损失
            loss = loss_fn(boxes, outputs)
            # 打印损失
            val_loss += loss.item()
            print("val_loss:%.3f, lr:%f, iteration:%d" % (val_loss / (iteration + 1), get_lr(optimizer), iteration))
            if math.isnan(val_loss):
                los_file.write("val_loss:%.3f, lr:%f, iteration:%d" % (val_loss / (iteration + 1), get_lr(optimizer), iteration))
                los_file.write("\r\n")
                if epoch > 50:
                    los_file.close() # 崩溃停止

    writer.add_scalar("valid_loss", val_loss / epoch_valid_step, epoch)
    print('Finish Validation')
    los_file.close()

    if (epoch + 1) % save_period == 0 or epoch + 1 == total_epoch:
        # 保存权值
        torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d_loss%.3f_val-loss%.3f.pth" % (
        epoch + 1, train_loss / epoch_train_step, val_loss / epoch_valid_step)))
