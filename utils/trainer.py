from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.utils.data.distributed
import torch.optim as optim
import numpy as np
import time
import os
from utils.my_dataset import MyDataset, MyDataLoader, InfiniteDataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
)


class Trainer:
    @staticmethod
    def create_datasets_with_k(features, shuffle=True, batch_size=None):
        """
        使用 features 构建 dataset
        :param features:
        :param choices_num: 选项(label) 个数
        :param shuffle: 是否随机顺序，默认 True
        :return:
        """
        if shuffle:
            perm = torch.randperm(len(features))
            features = [features[i] for i in perm]
        x = [i[0] for i in features]
        y = torch.tensor([i[1] for i in features])
        dataset = MyDataset(x, y)
        return dataset

    @staticmethod
    def create_datasets(features, shuffle=True, batch_size=None):
        """
        使用 features 构建 dataset
        :param features:
        :param choices_num: 选项(label) 个数
        :param shuffle: 是否随机顺序，默认 True
        :return:
        """
        if shuffle:
            perm = torch.randperm(len(features))
            features = [features[i] for i in perm]
        x = [i[0] for i in features]
        y = torch.tensor([i[1] for i in features])
        dataset = MyDataset(x, y)
        if batch_size is None:
            return dataset
        return MyDataLoader(dataset, batch_size=batch_size)

    @staticmethod
    def train(model, train_data, optimizer, scheduler, args):
        model.train()
        pbar = tqdm(train_data, ncols=80)
        # correct 代表累计正确率，count 代表目前已处理的数据个数
        correct = 0
        count = 0
        train_loss = 0.0
        pred_list = []
        accumulation_steps = args['accumulation_steps']
        for step, (x, y) in enumerate(pbar):
            # x, y = x.to(args['device']), y.to(args['device'])
            y = y.to(args['device'])
            output = model(x, labels=y)
            loss = output[0].mean()
            loss = loss / accumulation_steps
            loss.backward()
            if ((step + 1) % accumulation_steps) == 0:
                # 使用了梯度累积，实际batch_size等于args['batch_size']*accumulation_steps
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # 得到预测结果
            pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
            pred_list.append(output[1].softmax(dim=1))
            # print(output[1].softmax(dim=1))
            # 计算正确个数
            correct += pred.eq(y.view_as(pred)).sum().item()
            count += len(x)
            train_loss += loss.item()
            pbar.set_postfix({
                'loss': '{:.3f}'.format(loss.item()),
                'acc': '{:.3f}'.format(correct * 1.0 / count)
            })
            # gpu_track.track()
        pbar.close()
        return train_loss / count, correct * 1.0 / count, torch.cat(pred_list, dim=0)

    @staticmethod
    def test(model, test_data, args):
        model.eval()
        pbar = tqdm(test_data, ncols=100)
        test_loss = 0
        correct = 0
        count = 0
        pred_list = []
        with torch.no_grad():
            for step, (x, y) in enumerate(pbar):
                y = y.to(args['device'])
                output = model(x, labels=y)
                loss = output[0].mean()
                test_loss += loss.item()
                pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
                pred_list.append(output[1].softmax(dim=1))
                correct += pred.eq(y.view_as(pred)).sum().item()
                count += len(x)
                pbar.set_postfix({
                    'loss': '{:.3f}'.format(loss.item()),
                    'acc': '{:.3f}'.format(correct * 1.0 / count)
                })
        test_loss /= count
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
                test_loss, correct, count, 100. * correct / count))
        return test_loss, correct * 1.0 / count, torch.cat(pred_list, dim=0)

    @staticmethod
    def test_binary(model, test_data, args):
        model.eval()
        pbar = tqdm(test_data, ncols=100)
        test_loss = 0
        correct = 0
        count = 0
        pred_list = []
        pred_list2 = []
        sub_label_list = []
        with torch.no_grad():
            for step, (x, y) in enumerate(pbar):
                y = y.to(args['device'])
                output = model(x, labels=y)
                loss = output[0].mean()
                test_loss += loss.item()
                pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)

                sub_pred = output[1].softmax(dim=1)
                sub_pred2 = output[1]
                pred_list.append(sub_pred)
                pred_list2.append(sub_pred2)
                sub_label_list.append(y)

                correct += pred.eq(y.view_as(pred)).sum().item()
                count += len(x)

                pbar.set_postfix({
                    'loss': '{:.3f}'.format(loss.item()),
                    'acc': '{:.3f}'.format(correct * 1.0 / count)
                })
        test_loss /= count
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
                test_loss, correct, count, 100. * correct / count))

        all_pred = torch.cat(pred_list, dim=0).view(-1, 5, 2)
        all_pred2 = torch.cat(pred_list2, dim=0).view(-1, 5, 2)
        sub_label_list = torch.cat(sub_label_list, dim=0)
        real_label = sub_label_list.view(-1, 5).argmax(dim=1)
        acc = all_pred[:, :, 1].argmax(dim=1).eq(real_label).sum().item() / len(real_label)
        acc2 = all_pred2[:, :, 1].argmax(dim=1).eq(real_label).sum().item() / len(real_label)
        print("real accuracy: {}/{} ({:.4f}%)", all_pred[:, :, 1].argmax(dim=1).eq(real_label).sum().item(), len(real_label), acc)
        print("real accuracy: {}/{} ({:.4f}%)", all_pred2[:, :, 1].argmax(dim=1).eq(real_label).sum().item(), len(real_label), acc2)
        return test_loss, acc, all_pred

    @staticmethod
    def train_and_finetune(model, model_name, train_data1, train_data, test_data, args, subtask=2):
        global optimizer
        """
        因为涉及到固定网络部分层权重，目前没有手动设置而是采用在 model 初始化的时候设置，而在交叉验证中因为要多次创建 model，所以暂时将 model 的初始化放在这里
        """
        if args['model_init'] or model is None:
            model = model

        if subtask == 1:  # 如果是子任务1，改一下模型参数存放的目录
            args['checkpoints_dir'] = args['checkpoints_dir1']

        if args.get('use_wandb'):
            # 启用 wandb 记录实验数据
            import wandb
            wandb.init(project="kehetgat", config=args)
            wandb.watch(model)

        """手动固定网络除最后两层以外的所有，迫于无奈，先费点时间算出来需要 fix 多少层"""
        model_parameters = list(model.named_parameters())
        fix_idx = len(model_parameters) - args['unfixed_layer']
        white_list = ['classifier.weight', 'classifier.bias',
                      'electra.summary.summary.weight', 'electra.summary.summary.bias',
                      'electra.electra.summary.summary.weight', 'electra.electra.summary.summary.bias',
                      'electra.classifier.weight', 'electra.classifier.bias',
                      'electra.electra.classifier.weight', 'electra.electra.classifier.bias',
                      'fc3.weight', 'fc3.bias', 'project.weight', 'project.bias',
                      'project1.weight', 'project1.bias', 'project2.weight', 'project2.bias']
        print(type(model))
        print('unfixed layers: ', ', '.join(np.array(model_parameters)[fix_idx:, 0]))
        print('white list: ', ', '.join(white_list))
        for idx, (name, i) in enumerate(model.named_parameters()):
            # 这里测试得到 bert 本身前面 parameters 个数有 199 个
            if idx < fix_idx:
                i.requires_grad = False
            # 白名单的 layer 不进行 fix
            # print(name)
            if name in white_list:
                i.requires_grad = True

        model.to(args['device'])
        if torch.cuda.device_count() > 1 and args['use_multi_gpu']:
            print("{} GPUs are available. Let's use them.".format(
                torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
            # model = torch.nn.parallel.DistributedDataParallel(model,
            #                                                   device_ids=[0, 1],
            #                                                   output_device=0)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=args['lr'],
                               eps=args['adam_epsilon'],
                               weight_decay=0.01)
        num_train_steps = len(train_data) * args['epochs']
        num_warmup_steps = num_train_steps * args['warmup_proportion']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
        # gpu_track.track()

        acc = 0.0  # 准确率，以最高的一次为准，train_pred_opt 与 test_pred_opt 也是在准确率最高情况下算得
        train_pred_opt = None
        test_pred_opt = None
        writer = None
        if args['is_save_logs']:
            writer = SummaryWriter(os.path.join(args['logs_dir'], str(time.time())))

        start_time = time.time()
        # 先对预训练模型后几层进行训练
        print('start train...')
        for epoch in range(args['epochs']):
            print('Epoch {}/{}'.format(epoch + 1, args['epochs']))
            train_loss, train_acc, train_pred = Trainer.train(model, train_data1, optimizer, scheduler,
                                                              args)
            test_loss, test_acc, test_pred = Trainer.test(model, test_data, args)
            if test_acc > acc:
                # 在准确率最高的一次 finetune 中保存预测信息
                acc = test_acc
                train_pred_opt = train_pred
                test_pred_opt = test_pred
                # 保存效果最好的一次的权重，便于以后再利用
                torch.save(model.state_dict(),
                           os.path.join(args['checkpoints_dir'], 'best_model_{}_{}.pth'.format(model_name, args['exec_time'])))

            if writer is not None:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Loss/test', test_loss, epoch)
                writer.add_scalar('Accuracy/test', test_acc, epoch)

        for p in model.parameters():
            p.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=args['fine_tune_lr'], eps=args['adam_epsilon'],
                               weight_decay=args['weight_decay'])
        num_train_steps = len(train_data) * args['fine_tune_epochs']
        num_warmup_steps = num_train_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

        # 加载 train 时候的最优模型
        best_model_temp_path = os.path.join(args['checkpoints_dir'], 'best_model_{}_{}.pth'.format(model_name, args['exec_time']))
        if os.path.isfile(best_model_temp_path):
            print('load best model: {}'.format(best_model_temp_path))
            # 加载最优一次的权重
            model.load_state_dict(torch.load(
                best_model_temp_path
            ))

        # 整体进行 fine-tune
        print('start fine-tune...')
        for epoch in range(args['fine_tune_epochs']):
            print('Epoch {}/{}'.format(epoch + 1, args['fine_tune_epochs']))
            train_loss, train_acc, train_pred = Trainer.train(model, train_data, optimizer, scheduler,
                                                              args)
            test_loss, test_acc, test_pred = Trainer.test(model, test_data, args)
            if args.get('use_wandb'):
                import wandb
                wandb.log({"Train Accuracy": train_acc, "Train Loss": train_loss,
                           "Test Accuracy": test_acc, "Test Loss": test_loss})
            if test_acc > acc:
                # 在准确率最高的一次 finetune 中保存预测信息
                acc = test_acc
                train_pred_opt = train_pred
                test_pred_opt = test_pred
                # 保存效果最好的一次的权重，便于以后再利用
                torch.save(model.state_dict(),
                           os.path.join(args['checkpoints_dir'], 'best_model_{}_{}.pth'.format(model_name, args['exec_time'])))

            # 查找本次执行的最优预测结果为多少，保存最优权重
            if args['is_save_checkpoints']:
                pre_str = 'checkpoint_{}_{}_'.format(args['exec_time'], 'solo' if args['solo'] else 'pair')
                pth_list = [i for i in os.listdir(args['checkpoints_dir']) if i.startswith(pre_str)]
                max_acc = 0.0
                for i in pth_list:
                    acc_tmp = float(i.replace(pre_str, '').replace('.pth', ''))
                    max_acc = max(max_acc, acc_tmp)
                if acc * 100.0 > max_acc:
                    filename = 'checkpoint_{}_{}_{:06.3f}.pth'.format(args['exec_time'],
                                                                      'solo' if args['solo'] else 'pair',
                                                                      acc * 100.0)
                    checkpoint_path = os.path.join(args['checkpoints_dir'], filename)
                    torch.save(model.state_dict(), checkpoint_path)

            if writer is not None:
                writer.add_scalar('Loss/fine-tune train', train_loss, epoch)
                writer.add_scalar('Accuracy/fine-tune train', train_acc, epoch)
                writer.add_scalar('Loss/fine-tune test', test_loss, epoch)
                writer.add_scalar('Accuracy/fine-tune test', test_acc, epoch)
        print('Total Time: ', time.time() - start_time)

        if writer is not None:
            writer.close()

        best_model_temp_path = os.path.join(args['checkpoints_dir'], 'best_model_{}_{}.pth'.format(model_name, args['exec_time']))
        if os.path.isfile(best_model_temp_path):
            print('load best model: {}'.format(best_model_temp_path))
            # 加载最优一次的权重
            model.load_state_dict(torch.load(
                best_model_temp_path
            ))
        return acc, (train_pred_opt, test_pred_opt)

    @staticmethod
    def multi_train_and_finetune(model, model_name, train_datasets, test_data, args, subtask=2):
        global optimizer
        if args['model_init'] or model is None:
            model = model

        if subtask == 1:  # 如果是子任务1
            args['checkpoints_dir'] = args['checkpoints_dir1']

        """手动固定网络除最后两层以外的所有，迫于无奈，先费点时间算出来需要 fix 多少层"""
        model_parameters = list(model.named_parameters())
        # for x in model_parameters:
        #     print(x[0])
        fix_idx = len(model_parameters) - 4
        white_list = ['lamda1', 'lamda2']
        print(type(model))
        print('unfixed layers: ', ', '.join(np.array(model_parameters)[fix_idx:, 0]))
        print('white list: ', ', '.join(white_list))
        for idx, (name, i) in enumerate(model.named_parameters()):
            # 这里测试得到 bert 本身前面 parameters 个数有 199 个
            if idx < fix_idx:
                i.requires_grad = False
            # 白名单的 layer 不进行 fix
            if name in white_list:
                i.requires_grad = True

        model.to(args['device'])
        if torch.cuda.device_count() > 1 and args['use_multi_gpu']:
            print("{} GPUs are available. Let's use them.".format(
                torch.cuda.device_count()))
            # model = torch.nn.DataParallel(model)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[0, 1],
                                                              output_device=0)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=args['lr'],
                               eps=args['adam_epsilon'],
                               weight_decay=0.01)
        # gpu_track.track()

        acc = 0.0  # 准确率，以最高的一次为准，train_pred_opt 与 test_pred_opt 也是在准确率最高情况下算得
        train_pred_opt = None
        test_pred_opt = None

        start_time = time.time()

        train_iters = []
        tr_batches = []
        for idx, train_dataset in enumerate(train_datasets):
            # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'],
            #                                                sampler=train_sampler)
            # train_dataloader = MyDataLoader(train_dataset, batch_size=args['batch_size'])
            train_dataloader = train_dataset
            train_iters.append(InfiniteDataLoader(train_dataloader))
            tr_batches.append(len(train_dataloader))
        total_n_tr_batches = sum(tr_batches)
        sampling_prob = [float(i) / total_n_tr_batches for i in tr_batches]
        accumulation_steps = args['accumulation_steps']

        num_train_steps = total_n_tr_batches * args['epochs']
        num_warmup_steps = num_train_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

        # 先对预训练模型后几层进行训练
        print('start train...')
        for epoch in range(args['epochs']):
            print('Epoch {}/{}'.format(epoch + 1, args['epochs']))
            model.train()
            pbar = tqdm(range(total_n_tr_batches), ncols=100)
            # correct 代表累计正确率，count 代表目前已处理的数据个数
            correct = 0
            count = 0
            train_loss = 0.0
            pred_list = []
            task_id = 0
            for step, _ in enumerate(pbar):
                task_id = np.argmax(np.random.multinomial(1, sampling_prob))
                batch = train_iters[task_id].get_next()
                x = batch[0]
                y = batch[1].to(args['device'])
                # optimizer.zero_grad()
                output = model(x, labels=y, task_id=task_id)
                loss = output[0].mean()
                loss = loss / accumulation_steps
                loss.backward()
                if ((step + 1) % accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                # 得到预测结果
                pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
                pred_list.append(output[1].softmax(dim=1))
                # print(output[1].softmax(dim=1))
                # 计算正确个数
                correct += pred.eq(y.view_as(pred)).sum().item()
                count += len(x)
                train_loss += loss.item()
                pbar.set_postfix({
                    'loss': '{:.3f}'.format(loss.item()),
                    'acc': '{:.3f}'.format(correct * 1.0 / count)
                })
            pbar.close()
            # train_loss, train_acc = train_loss / count, correct * 1.0 / count
            # train_loss, train_acc, train_pred = Trainer.train(model, train_data, optimizer, scheduler,
            #                                                   args)

            test_loss, test_acc, test_pred = Trainer.test(model, test_data, args)
            if test_acc > acc:
                # 在准确率最高的一次 finetune 中保存预测信息
                acc = test_acc
                # train_pred_opt = train_pred
                # test_pred_opt = test_pred
                # 保存效果最好的一次的权重，便于以后再利用
                torch.save(model.state_dict(),
                           os.path.join(args['checkpoints_dir'], 'best_model_{}_{}.pth'.format(model_name, args['exec_time'])))

        for p in model.parameters():
            p.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=args['fine_tune_lr'], eps=args['adam_epsilon'],
                               weight_decay=args['weight_decay'])
        num_train_steps = total_n_tr_batches * args['fine_tune_epochs']
        num_warmup_steps = num_train_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

        # 加载 train 时候的最优模型
        best_model_temp_path = os.path.join(args['checkpoints_dir'], 'best_model_{}_{}.pth'.format(model_name, args['exec_time']))
        if os.path.isfile(best_model_temp_path):
            print('load best model: {}'.format(best_model_temp_path))
            # 加载最优一次的权重
            model.load_state_dict(torch.load(
                best_model_temp_path
            ))

        # 整体进行 fine-tune
        print('start fine-tune...')
        for epoch in range(args['fine_tune_epochs']):
            print('Epoch {}/{}'.format(epoch + 1, args['fine_tune_epochs']))
            pbar = tqdm(range(total_n_tr_batches), ncols=100)
            # correct 代表累计正确率，count 代表目前已处理的数据个数
            correct = 0
            count = 0
            train_loss = 0.0
            pred_list = []
            task_id = 0
            for step, _ in enumerate(pbar):
                model.train()
                task_id = np.argmax(np.random.multinomial(1, sampling_prob))
                batch = train_iters[task_id].get_next()
                x = batch[0]
                y = batch[1].to(args['device'])
                # optimizer.zero_grad()
                output = model(x, labels=y, task_id=task_id)
                loss = output[0].mean()
                loss = loss / accumulation_steps
                loss.backward()
                if ((step + 1) % accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                # 得到预测结果
                pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
                pred_list.append(output[1].softmax(dim=1))
                # print(output[1].softmax(dim=1))
                # 计算正确个数
                correct += pred.eq(y.view_as(pred)).sum().item()
                count += len(x)
                train_loss += loss.item()
                pbar.set_postfix({
                    'loss': '{:.3f}'.format(loss.item()),
                    'acc': '{:.3f}'.format(correct * 1.0 / count)
                })
                # if (step + 1) % 1000 == 0:
                #     test_loss, test_acc, test_pred = Trainer.test(model, test_data, args)
                #     if test_acc > acc:
                #         # 在准确率最高的一次 finetune 中保存预测信息
                #         acc = test_acc
                #         # train_pred_opt = train_pred
                #         # test_pred_opt = test_pred
                #         # 保存效果最好的一次的权重，便于以后再利用
                #         torch.save(model.state_dict(),
                #                    os.path.join(args['checkpoints_dir'],
                #                                 'best_model_temp_{}.pth'.format(args['exec_time'])))
            pbar.close()
            # train_loss, train_acc, train_pred = train_loss / count, correct * 1.0 / count, torch.cat(pred_list, dim=0)
            test_loss, test_acc, test_pred = Trainer.test(model, test_data, args)
            # if args.get('use_wandb'):
            #     import wandb
            #     wandb.log({"Train Accuracy": train_acc, "Train Loss": train_loss,
            #                "Test Accuracy": test_acc, "Test Loss": test_loss})
            if test_acc > acc:
                # 在准确率最高的一次 finetune 中保存预测信息
                acc = test_acc
                # train_pred_opt = train_pred
                # test_pred_opt = test_pred
                # 保存效果最好的一次的权重，便于以后再利用
                torch.save(model.state_dict(),
                           os.path.join(args['checkpoints_dir'], 'best_model_{}_{}.pth'.format(model_name, args['exec_time'])))

            # 查找本次执行的最优预测结果为多少，保存最优权重
            if args['is_save_checkpoints']:
                pre_str = 'checkpoint_{}_{}_'.format(args['exec_time'], 'solo' if args['solo'] else 'pair')
                pth_list = [i for i in os.listdir(args['checkpoints_dir']) if i.startswith(pre_str)]
                max_acc = 0.0
                for i in pth_list:
                    acc_tmp = float(i.replace(pre_str, '').replace('.pth', ''))
                    max_acc = max(max_acc, acc_tmp)
                if acc * 100.0 > max_acc:
                    filename = 'checkpoint_{}_{}_{:06.3f}.pth'.format(args['exec_time'],
                                                                      'solo' if args['solo'] else 'pair',
                                                                      acc * 100.0)
                    checkpoint_path = os.path.join(args['checkpoints_dir'], filename)
                    torch.save(model.state_dict(), checkpoint_path)

        print('Total Time: ', time.time() - start_time)

        best_model_temp_path = os.path.join(args['checkpoints_dir'], 'best_model_{}_{}.pth'.format(model_name, args['exec_time']))
        if os.path.isfile(best_model_temp_path):
            print('load best model: {}'.format(best_model_temp_path))
            # 加载最优一次的权重
            model.load_state_dict(torch.load(
                best_model_temp_path
            ))
        return acc, (train_pred_opt, test_pred_opt)

    @staticmethod
    def train_only(model, train_data, test_data, args):
        if args['use_multi_gpu'] and torch.cuda.device_count() > 1:
            print("{} GPUs are available. Let's use them.".format(
                torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
        model.to(args['device'])

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args['lr'], eps=args['adam_epsilon'])
        num_train_steps = len(train_data) * args['epochs']
        num_warmup_steps = num_train_steps * 0.05
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

        print('start train...')
        print(type(model))

        acc = 0
        for epoch in range(args['epochs']):
            print('Epoch {}/{}'.format(epoch + 1, args['epochs']))
            train_loss, train_acc, train_pred = Trainer.train(model, train_data, optimizer, scheduler, args)
            test_loss, test_acc, test_pred = Trainer.test(model, test_data, args)

            if test_acc > acc:
                acc = test_acc
        print('final acc: ', acc)

    @staticmethod
    def test_only(model, test_data, args):
        model.eval()
        pbar = tqdm(test_data, ncols=100)
        test_loss = 0
        correct = 0
        count = 0
        pred_list = []
        pred_error_list = []  # 预测错误的数据列表
        dev_index = 0
        pred_label = []
        real_label = []
        with torch.no_grad():
            for step, (x, y) in enumerate(pbar):
                # x, y = x.to(args['device']), y.to(args['device'])
                y = y.to(args['device'])
                output = model(x, labels=y)
                loss = output[0].mean()
                test_loss += loss.item()
                pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
                pred_list.append(output[1].softmax(dim=1))
                pred_label.append(output[1].softmax(dim=1).argmax(dim=1))
                real_label.append(y)
                tt = pred.eq(y.view_as(pred)).cpu().numpy().squeeze()
                tt = np.argwhere(tt == 0).reshape(-1)
                tt = dev_index * args['test_batch_size'] + tt
                pred_error_list.extend([i for i in tt])
                correct += pred.eq(y.view_as(pred)).sum().item()
                count += len(x)
                dev_index += 1
                pbar.set_postfix({
                    'loss': '{:.3f}'.format(loss.item()),
                    'acc': '{:.3f}'.format(correct * 1.0 / count)
                })
        test_loss /= count
        pred_label_list = torch.cat(pred_label, dim=0).cpu().numpy()
        real_label_list = torch.cat(real_label, dim=0).cpu().numpy()
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
                test_loss, correct, count, 100. * correct / count))
        return test_loss, correct * 1.0 / count, torch.cat(pred_list, dim=0), pred_error_list, pred_label_list, real_label_list

    @staticmethod
    def fine_tune_only(model, model_name, train_data, test_data, args, subtask=2):
        if subtask == 1:  # 如果是子任务1
            args['checkpoints_dir'] = args['checkpoints_dir1']

        acc = 0.0
        train_pred_opt = None
        test_pred_opt = None
        for p in model.parameters():
            p.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=args['fine_tune_lr'], eps=args['adam_epsilon'],
                               weight_decay=args['weight_decay'])
        num_train_steps = len(train_data) * args['fine_tune_epochs']
        num_warmup_steps = num_train_steps * 0.05
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

        # 整体进行 fine-tune
        print('start fine-tune...')
        for epoch in range(args['fine_tune_epochs']):
            print('Epoch {}/{}'.format(epoch + 1, args['fine_tune_epochs']))
            train_loss, train_acc, train_pred = Trainer.train(model, train_data, optimizer, scheduler,
                                                              args)
            test_loss, test_acc, test_pred = Trainer.test(model, test_data, args)
            if args.get('use_wandb'):
                import wandb
                wandb.log({"Train Accuracy": train_acc, "Train Loss": train_loss,
                           "Test Accuracy": test_acc, "Test Loss": test_loss})
            if test_acc > acc:
                # 在准确率最高的一次 finetune 中保存预测信息
                acc = test_acc
                train_pred_opt = train_pred
                test_pred_opt = test_pred
                # 保存效果最好的一次的权重，便于以后再利用
                torch.save(model.state_dict(),
                           os.path.join(args['checkpoints_dir'], 'best_model_{}_{}.pth'.format(model_name, args['exec_time'])))

            # 查找本次执行的最优预测结果为多少，保存最优权重
            if args['is_save_checkpoints']:
                pre_str = 'checkpoint_{}_{}_'.format(args['exec_time'], 'solo' if args['solo'] else 'pair')
                pth_list = [i for i in os.listdir(args['checkpoints_dir']) if i.startswith(pre_str)]
                max_acc = 0.0
                for i in pth_list:
                    acc_tmp = float(i.replace(pre_str, '').replace('.pth', ''))
                    max_acc = max(max_acc, acc_tmp)
                if acc * 100.0 > max_acc:
                    filename = 'checkpoint_{}_{}_{:06.3f}.pth'.format(args['exec_time'],
                                                                      'solo' if args['solo'] else 'pair',
                                                                      acc * 100.0)
                    checkpoint_path = os.path.join(args['checkpoints_dir'], filename)
                    torch.save(model.state_dict(), checkpoint_path)

        best_model_temp_path = os.path.join(args['checkpoints_dir'], 'best_model_{}_{}.pth'.format(model_name, args['exec_time']))
        if os.path.isfile(best_model_temp_path):
            print('load best model: {}'.format(best_model_temp_path))
            # 加载最优一次的权重
            model.load_state_dict(torch.load(
                best_model_temp_path
            ))
        return acc, (train_pred_opt, test_pred_opt)

    @staticmethod
    def test_SVM(model, test_data, args):
        model.eval()
        pbar = tqdm(test_data, ncols=100)
        test_loss = 0
        correct = 0
        count = 0
        pred_list = []
        svm_feature_list = []
        new_label_list = []
        with torch.no_grad():
            for step, (x, y) in enumerate(pbar):
                y = y.to(args['device'])

                new_y = y.unsqueeze(1)
                new_y = torch.zeros(len(new_y), 5).cuda().scatter_(1, new_y, 1).view(-1).long()  # 改为二分类
                output = model(x, labels=y)
                svm_feature = output[2]

                loss = output[0].mean()
                test_loss += loss.item()
                pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
                pred_list.append(output[1].softmax(dim=1))

                correct += pred.eq(y.view_as(pred)).sum().item()
                count += len(x)

                svm_feature_list.append(svm_feature)
                new_label_list.append(new_y)

                pbar.set_postfix({
                    'loss': '{:.3f}'.format(loss.item()),
                    'acc': '{:.3f}'.format(correct * 1.0 / count)
                })
            pbar.close()
        test_loss /= count
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
                test_loss, correct, count, 100. * correct / count))
        svm_feature_list = torch.cat(svm_feature_list, dim=0).cpu().numpy()
        new_label_list = torch.cat(new_label_list, dim=0).cpu().numpy()

        return svm_feature_list, new_label_list
