import datetime
import os
import time

import torch

import gc
gc.collect()
torch.cuda.empty_cache()

import torch.utils.data
from torch import nn
import torchvision

import utils

try:
    from apex import amp
except ImportError:
    amp = None

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.transforms import autoaugment, transforms

import pandas as pd

import numpy as np
from itertools import cycle

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from statistics import mean


n_classes = 3








def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(",")
            imlist.append((impath, int(imlabel)))
    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def main(args):
    device = torch.device("cuda")
    #torch.backends.cudnn.benchmark = True

    start_time = time.time()
    k = 1
    while k < 6:
        print("Load {}-fold dataset".format(k))

        total = 11977
        path_data = "/home/jkim/Project/references/classification/data/" + str(k)

        path_output = "/home/jkim/Project/references/classification/output/" + str(k)
        os.makedirs(path_output, exist_ok=True)

        train_accuracy = []
        train_loss = []

        test_accuracy = []
        test_loss = []

        roc_actual = []
        roc_predicted = []

        # resize_size, crop_size = (256, 256)
        # resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (256, 224)
        resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (256, 224)
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        hflip_prob = 0.5

        trans = [transforms.RandomResizedCrop(crop_size)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
            trans.append(autoaugment.AutoAugment(policy=aa_policy))
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=path_data, flist=path_data + "/train.csv", transform=transforms.Compose(trans)),
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=path_data, flist=path_data + "/test.csv", transform=transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])),
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=True)

        loaded = len(train_loader.dataset) + len(test_loader.dataset)

        if total != loaded:
            print("ERROR: Failed to verify the count of file.")
            exit(1)

        print("Loaded {} files".format(loaded))
        print("Creating model")

        model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)
        model.to(device)

        criterion = nn.CrossEntropyLoss()

        opt_name = args.opt.lower()
        if opt_name == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif opt_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
        else:
            raise RuntimeError("Invalid optimizer {}. Only SGD and RMSprop are supported.".format(args.opt))

        if args.apex:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level=args.apex_opt_level
                                              )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

        model_without_ddp = model

        print("Start training")

        if (len(train_accuracy) != 0) or (len(train_loss) != 0) or (len(test_accuracy) != 0) or (len(test_loss) != 0) or (len(roc_actual) != 0) or (len(roc_predicted) != 0):
            print("ERROR: Failed to initialze metric lists.")
            exit(1)

        print_freq = 10
        for epoch in range(args.start_epoch, args.epochs):
            model.train()
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
            metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

            header = 'Epoch: [{}]'.format(epoch)
            for image, target in metric_logger.log_every(train_loader, print_freq, header):
                start_time = time.time()
                image, target = image.to(device), target.to(device)
                output, aux2 = model(image)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
                metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

                # conf_matrix = confusion_matrix(output, target, conf_matrix)

            train_accuracy.append(float('{top1.global_avg:.3f}'.format(top1=metric_logger.acc1)))
            train_loss.append(float('{loss:.3f}'.format(loss=loss.item())))
            print(train_accuracy)
            print(train_loss)

            lr_scheduler.step()
            epoch_final = epoch

        print('Training process has finished. Saving trained model.')

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch_final,
            'args': args}
        utils.save_on_master(
            checkpoint,
            os.path.join(path_output, 'model_{}.pth'.format(epoch_final)))

        print('Starting evaluating')

        model.eval()
        val_loss = 0
        val_correct = 0
        print_freq = 100

        criterion = nn.CrossEntropyLoss()

        pred_list = torch.Tensor([]).to(device).long()
        target_list = torch.Tensor([]).to(device).long()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        with torch.no_grad():
            for image, target in metric_logger.log_every(test_loader, print_freq, header):
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(image)
                loss = criterion(output, target)

                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        print("\n########## {}-Fold Model Evaluation Complete ##########".format(k))
        print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

        # Log loss
        val_loss += criterion(output, target.long()).item()

        # Calculate the number of correctly classified examples
        pred = output.argmax(dim=1, keepdim=True)
        val_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Bookkeeping
        pred_list = torch.cat([pred_list, pred.squeeze()])
        target_list = torch.cat([target_list, target.squeeze()])

        roc_actual = []
        roc_predicted = []

        roc_actual.extend(target_list.tolist())
        roc_predicted.extend(pred_list.tolist())
        print(roc_actual)
        print(roc_predicted)

        test_accuracy.append(float('{top1.global_avg:.3f}'.format(top1=metric_logger.acc1)))
        test_loss.append(float('{loss:.3f}'.format(loss=loss.item())))
        print(test_accuracy)
        print(test_loss)

        print(len(roc_actual), "actual\t\t\t", roc_actual)
        print(len(roc_predicted), "predicted\t\t", roc_predicted)
        print(len(train_accuracy), "train_accuracy\t", train_accuracy)
        print(len(train_loss), "train_loss\t\t", train_loss)
        print(len(test_accuracy), "test_accuracy\t", test_accuracy)
        print(len(test_loss), "test_loss\t\t", test_loss)

        # Plot Accuracy
        plt.plot([i for i in range(1, len(train_accuracy) + 1)], train_accuracy)
        #plt.plot([i for i in range(1, len(test_accuracy) + 1)], test_accuracy)
        plt.axis([1, len(train_loss) + 1, min(train_loss) - 0.1, 1])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        #plt.legend(['Train', 'Test'], loc='upper left')
        plt.legend(['Train'], loc='upper left')
        plt.savefig(path_output + "/accuracy.png")
        #plt.show()

        # Plot Loss
        plt.plot([i for i in range(1, len(train_loss) + 1)], train_loss)
        #plt.plot([i for i in range(1, len(test_loss) + 1)], test_loss)
        #plt.axis([1, len(train_loss) + 1, min(train_loss + test_loss) - 0.1, 1])
        plt.axis([1, len(train_loss) + 1, min(train_loss) - 0.1, 1])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.savefig(path_output + "/loss.png")
        #plt.show()

        # Plot ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        actual_np = np.array(pd.get_dummies(roc_actual))
        predicted_np = np.array(pd.get_dummies(roc_predicted))

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(actual_np[:, i], predicted_np[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(actual_np.ravel(), predicted_np.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        lw = 2
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

        labels = ["ADIMUC", "STRMUS", "TUMSTU"]
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} {1} (area = {2:0.2f})'
                           ''.format(i, labels[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig(path_output + "/roc.png")
        #plt.show()

        # Classification Evaluation Metrics
        report = classification_report(roc_actual, roc_predicted, target_names=labels, output_dict=True)
        matrix = confusion_matrix(roc_actual, roc_predicted)

        print("\n\tClassification Evaluation Metrics\n")
        print("\t\t\t\t\t{:s}\t{:s}\t{:s}\tALL".format(labels[0], labels[1], labels[2]))
        print("Precision\t\t\t{:.3f}\t{:.3f}\t{:.3f}".format(report[labels[0]]["precision"],
                                                             report[labels[1]]["precision"],
                                                             report[labels[2]]["precision"]))
        print("Recall(sensitivity)\t{:.3f}\t{:.3f}\t{:.3f}".format(report[labels[0]]["precision"],
                                                                   report[labels[1]]["precision"],
                                                                   report[labels[2]]["precision"]))
        print("Specificity\t\t\t{:.3f}\t{:.3f}\t{:.3f}".format(matrix[0][0] / report[labels[0]]["support"],
                                                               matrix[1][1] / report[labels[1]]["support"],
                                                               matrix[2][2] / report[labels[2]]["support"]))
        print("AUC\t\t\t\t\t{:.3f}\t{:.3f}\t{:.3f}".format(roc_auc[0], roc_auc[1], roc_auc[2]))
        print("Accuracy\t\t\t\t\t\t\t\t\t{:.3f}".format(report["accuracy"]))

        k += 1


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training', add_help=add_help)

    parser.add_argument('--data-path', default='/home/jkim/Project/references/classification/data', help='dataset')
    parser.add_argument('--model', default='inception_v3', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='output', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--auto-augment', default=None, help='auto augment policy (default: None)')
    parser.add_argument('--random-erase', default=0.0, type=float, help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
