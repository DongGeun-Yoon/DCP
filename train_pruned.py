import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from config import device, im_size, grad_clip, print_freq
from data_gen import DIMDataset
from models import DIMModel, DIMModel_student
from utils import *
from KD_loss import *

def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best, dir='normal'):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    root = os.path.join('result', dir)
    ensure_folder(root)
    torch.save(state, os.path.join(root, 'pruned_{}.tar'.format(epoch+1)))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(root,'BEST_pruned.tar'))

def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0
    decays_since_improvement = 0

    # teacher setting
    teacher = torch.load('./pretrained/BEST_checkpoint.tar')
    teacher_model = teacher['model'].module
    for p in teacher_model.parameters():
        p.requires_grad = False

    cfg = torch.load(os.path.join('result', args.save_dir, args.pruned_cfg))['cfg']
    # load checkpoint
    if checkpoint == '':
        model = DIMModel_student(n_classes=1, in_channels=4, is_unpooling=True, cfg=cfg)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model'].module
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    train_dataset = DIMDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_dataset = DIMDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
            
    # Epochs
    for epoch in range(start_epoch, args.pruned_epoch):
        if args.optimizer == 'sgd' and epochs_since_improvement == 40:
            break

        if args.optimizer == 'sgd' and epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            checkpoint = 'BEST_checkpoint.tar'
            checkpoint = torch.load(checkpoint)
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            decays_since_improvement += 1
            print("\nDecays since last improvement: %d\n" % (decays_since_improvement,))
            adjust_learning_rate(optimizer, 0.6 ** decays_since_improvement)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                            teacher_model=teacher_model,
                            student_model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            logger=logger,
                            args=args)
        effective_lr = get_learning_rate(optimizer)
        print('Current effective learning rate: {}\n'.format(effective_lr))

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Learning_Rate', effective_lr, epoch)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           logger=logger)

        writer.add_scalar('Valid_Loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            decays_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best, dir=args.KD)


def train(train_loader, teacher_model, student_model, optimizer, epoch, logger, args):
    student_model.train()  # train mode (dropout and batchnorm is used)
    losses = AverageMeter()

    # select kd
    if args.KD == 'SPKD':
        kd = SPKD().to(device)
    elif args.KD == 'NST':
        kd = NST().to(device)
    elif args.KD == 'OFD':
        kd = Distiller().to(device)
    # Batches
    for i, (img, alpha_label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(device)  # [N, 4, 320, 320]
        alpha_label = alpha_label.type(torch.FloatTensor).to(device)  # [N, 320, 320]
        alpha_label = alpha_label.reshape((-1, 2, im_size * im_size))  # [N, 320*320]
            
        # Forward prop.
        with torch.no_grad():
            teacher_fms, teacher_out = teacher_model(img)  # [N, 3, 320, 320]
        student_fms, student_out = student_model(img)  # [N, 3, 320, 320]

        student_out = student_out.reshape((-1, 1, im_size * im_size))  # [N, 320*320]
        teacher_out = teacher_out.reshape((-1, 1, im_size * im_size))  # [N, 320*320]

        # Calculate loss
        loss = alpha_prediction_loss(student_out, alpha_label)

        # KD loss
        if args.KD == 'SPKD' or args.KD == 'NST':
            kd_loss = kd(student_fms, teacher_fms)
        elif args.KD == 'OFD':
            kd_loss = kd(img).sum() / 1000
        TS_loss = alpha_prediction_loss(student_out, teacher_out, alpha_label[:, 1, :])

        loss = loss + TS_loss + kd_loss
        # Back prop.
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses)
            logger.info(status)

    return losses.avg


def valid(valid_loader, model, logger):
    losses = AverageMeter()
    model.eval()  # eval mode (dropout and batchnorm is NOT used)
    # Batches
    for img, alpha_label in valid_loader:
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(device)  # [N, 3, 320, 320]
        alpha_label = alpha_label.type(torch.FloatTensor).to(device)  # [N, 320, 320]
        alpha_label = alpha_label.reshape((-1, 2, im_size * im_size))  # [N, 320*320]

        # Forward prop.
        _, alpha_out = model(img)  # [N, 320, 320]
        alpha_out = alpha_out.reshape((-1, 1, im_size * im_size))  # [N, 320*320]

        # Calculate loss
        loss = alpha_prediction_loss(alpha_out, alpha_label)

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    status = 'Validation: Loss {loss.avg:.4f}\n'.format(loss=losses)
    logger.info(status)
    return losses.avg

def main():
    parser = argparse.ArgumentParser(description='Train Pruned network')
    parser.add_argument('--config', type=str, default="configs/train_SPKD.yaml", help="Path to yaml config file")
    args = parser.parse_args()

    args = get_config(args.config)
    train_net(args)

if __name__ == '__main__':
    main()
