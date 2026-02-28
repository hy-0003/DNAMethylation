import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module import Adversarial_module, Fusion_module
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score


class ModelManager():
    def __init__(self, learner):
        self.learner = learner
        self.IOManager = learner.IOManager
        self.visualizer = learner.visualizer
        self.dataManager = learner.dataManager
        self.config = learner.config

        self.model = None
        self.optimizer = None
        self.loss_func = None

        # supervised learning
        self.best_performance = None
        self.best_repres_list = None
        self.best_label_list = None
        self.test_performance = []
        self.valid_performance = []
        self.avg_test_loss = 0

    def init_model(self):
        self.model = Fusion_module.Fusion(self.config)
        if self.config.cuda:
            self.model.cuda()

    def load_params(self):
        if self.config.path_params:
            self.model = self.__load_params(self.model, self.config.path_params)

    def adjust_model(self):
        self.model = self.__adjust_model(self.model)

    def init_optimizer(self):
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config.lr,
                                               weight_decay=self.config.reg)

    def def_loss_func(self):
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        train_dataloader = self.dataManager.get_dataloder(name='train_set')
        test_dataloader = self.dataManager.get_dataloder(name='test_set')
        best_performance, best_repres_list, best_label_list, ROC, PRC = self.__SL_train(train_dataloader,test_dataloader)
                
        self.visualizer.roc_data = ROC
        self.visualizer.prc_data = PRC
        self.visualizer.repres_list = best_repres_list
        self.visualizer.label_list = best_label_list

        self.best_repres_list = best_repres_list
        self.best_label_list = best_label_list
        self.best_performance = best_performance
        self.IOManager.log.Info('Best Performance: {}'.format(self.best_performance))
        self.IOManager.log.Info('Performance: {}'.format(self.test_performance))


    def test(self):
        self.model.eval()
        test_dataloader = self.dataManager.get_dataloder('test_set')
        if test_dataloader is not None:
            test_performance, avg_test_loss, ROC_data, PRC_data, repres_list, label_list = self.__SL_test(test_dataloader)
            log_text = '\n' + '=' * 20 + ' Final Test Performance ' + '=' * 20 \
                           + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                test_performance[0], test_performance[1], test_performance[2], test_performance[3],
                test_performance[4]) \
                           + '\n' + '=' * 60
            self.IOManager.log.Info(log_text)
        else:
            self.IOManager.log.Warn('Test Data is None.')


    def __load_params(self, model, param_path):
        pretrained_dict = torch.load(param_path)
        new_model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        model.load_state_dict(new_model_dict)
        return model

    def __adjust_model(self, model):
        '''
        print('-' * 50, 'Model.named_parameters', '-' * 50)
        for name, value in model.named_parameters():
            print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))
        '''
        # Count the total parameters
        params = list(model.parameters())
        k = 0
        for i in params:
            l = 1
            for j in i.size():
                l *= j
            k = k + l
        print('=' * 50, "Number of total parameters:" + str(k), '=' * 50)
        return model

    def __get_loss(self, logits, label):
        loss = 0
        loss = self.loss_func(logits.view(-1, self.config.num_class), label.view(-1))
        loss = (loss.float()).mean()
            # flooding method
        loss = (loss - self.config.b).abs() + self.config.b
        return loss

    def __caculate_metric(self, pred_prob, label_pred, label_real):
        test_num = len(label_real)
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for index in range(test_num):
            if label_real[index] == 1:
                if label_real[index] == label_pred[index]:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if label_real[index] == label_pred[index]:
                    tn = tn + 1
                else:
                    fp = fp + 1

        # Accuracy
        ACC = float(tp + tn) / test_num

        # Precision
        # if tp + fp == 0:
        #     Precision = 0
        # else:
        #     Precision = float(tp) / (tp + fp)

        # Sensitivity
        if tp + fn == 0:
            Recall = Sensitivity = 0
        else:
            Recall = Sensitivity = float(tp) / (tp + fn)

        # Specificity
        if tn + fp == 0:
            Specificity = 0
        else:
            Specificity = float(tn) / (tn + fp)

        # MCC
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
            MCC = 0
        else:
            MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

        # F1-score
        # if Recall + Precision == 0:
        #     F1 = 0
        # else:
        #     F1 = 2 * Recall * Precision / (Recall + Precision)

        # ROC and AUC
        FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)  # Default 1 is positive sample
        AUC = auc(FPR, TPR)

        # PRC and AP
        precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
        AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

        # ROC(FPR, TPR, AUC)
        # PRC(Recall, Precision, AP)

        performance = [ACC, Sensitivity, Specificity, AUC, MCC]
        roc_data = [FPR, TPR, AUC]
        prc_data = [recall, precision, AP]
        return performance, roc_data, prc_data


    def __SL_train(self, train_dataloader, test_dataloader):
        step = 0
        best_mcc = 0
        best_performance = None
        best_ROC = None
        best_PRC = None
        best_repres_list = None
        best_label_list = None

        if self.config.adversarial == True:
            FGMModel = Adversarial_module.FGM(self.model, 'bertone.bert.embeddings.word_embeddings', 'berttwo.bert.embeddings.word_embeddings')

        for epoch in range(1, self.config.epoch + 1):
            self.model.train()
            for batch in train_dataloader:
                data, label = batch
                logits, _ = self.model(data)
                train_loss = self.__get_loss(logits, label)
                self.optimizer.zero_grad()
                train_loss.backward()


                if self.config.adversarial == True:
                    FGMModel.attack()
                    # self.optimizer.zero_grad()  # 如果不想累加梯度，就把这里的注释取消
                    logitsattack, _ = self.model(data)
                    attack_train_loss = self.__get_loss(logitsattack, label)
                    attack_train_loss.backward()

                    FGMModel.restore()
                self.optimizer.step()
                step += 1

                '''Periodic Train Log'''
                if step % self.config.interval_log == 0:
                    corrects = (torch.max(logits, 1)[1] == label).sum()
                    the_batch_size = label.shape[0]
                    train_acc = 100.0 * corrects / the_batch_size
                    print('Epoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, step, train_loss,
                                                                                            train_acc,
                                                                                            corrects,
                                                                                            the_batch_size))
                    self.visualizer.step_log_interval.append(step)

                    if self.config.cuda:
                        self.visualizer.train_metric_record.append(train_acc.cpu().detach().numpy())
                        self.visualizer.train_loss_record.append(train_loss.cpu().detach().numpy())
                    else:
                        self.visualizer.train_metric_record.append(train_acc.cpu().detach().numpy())
                        self.visualizer.train_loss_record.append(train_loss.cpu().detach().numpy())

            '''Periodic Test'''
            if epoch % self.config.interval_test == 0:
                self.model.eval()
                test_performance, avg_test_loss, ROC_data, PRC_data, repres_list, label_list = self.__SL_test(
                    test_dataloader)
                self.visualizer.step_test_interval.append(epoch)
                self.visualizer.test_metric_record.append(test_performance[0])
                self.visualizer.test_loss_record.append(avg_test_loss.cpu().detach().numpy())
                self.test_performance.append(test_performance)

                log_text = '\n' + '=' * 20 + ' Test Performance. Epoch[{}] '.format(epoch) + '=' * 20 \
                           + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                    test_performance[0], test_performance[1], test_performance[2], test_performance[3],
                    test_performance[4]) \
                           + '\n' + '=' * 60
                self.IOManager.log.Info(log_text)

                test_mcc = test_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
                if test_mcc > best_mcc:
                    best_mcc = test_mcc
                    best_performance = test_performance
                    best_ROC = ROC_data
                    best_PRC = PRC_data
                    best_repres_list = repres_list
                    best_label_list = label_list
                    if self.config.save_best and best_mcc > self.config.threshold:
                        self.IOManager.save_model_dict(self.model.state_dict(), self.config.model_save_name,
                                                       'ACC', best_mcc)
        return best_performance, best_repres_list, best_label_list, best_ROC, best_PRC
    

    def __SL_test(self, dataloader):
        corrects = 0
        test_batch_num = 0
        test_sample_num = 0
        avg_test_loss = 0
        pred_prob = []
        label_pred = []
        label_real = []

        repres_list = []
        label_list = []

        with torch.no_grad():
            for batch in dataloader:
                data, label = batch
                logits, representation = self.model(data)
                avg_test_loss += self.__get_loss(logits, label)

                repres_list.extend(representation.cpu().detach().numpy())
                label_list.extend(label.cpu().detach().numpy())

                pred_prob_all = F.softmax(logits, dim=1)  # [batch_size, class_num]
                pred_prob_positive = pred_prob_all[:, 1]  
                pred_prob_sort = torch.max(pred_prob_all, 1)  # [batch_size]
                pred_class = pred_prob_sort[1]  # [batch_size]

                corrects += (pred_class == label).sum()
                test_sample_num += len(label)
                test_batch_num += 1
                pred_prob = pred_prob + pred_prob_positive.tolist()
                label_pred = label_pred + pred_class.tolist()
                label_real = label_real + label.tolist()

        performance, ROC_data, PRC_data = self.__caculate_metric(pred_prob, label_pred, label_real)

        avg_test_loss /= test_batch_num
        avg_acc = 100.0 * corrects / test_sample_num
        print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_test_loss,
                                                                      avg_acc,
                                                                      corrects,
                                                                      test_sample_num))

        self.avg_test_loss = avg_test_loss
        return performance, avg_test_loss, ROC_data, PRC_data, repres_list, label_list
