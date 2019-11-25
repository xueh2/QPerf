
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import os
import sys
from tensorboardX import SummaryWriter
from .performance import *
from utils import *
from time import time
from IPython.display import clear_output, display
from .post_process import *

'''
Base class for training
'''
class GadgetronTrainer(object):
    def __init__(self, model, optimizer, criterion, loader_for_train, loader_for_val, scheduler=None, epochs=10, device=torch.device('cpu'), x_dtype=torch.float32, y_dtype=torch.long,  early_stopping_thres=10, print_every=100, writer = None, run_id_str="training/", experiment=None):
        r"""
        Define the trainer class for gadgetron NN
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.loader_for_train = loader_for_train
        self.loader_for_val = loader_for_val
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.early_stopping_thres = early_stopping_thres
        self.print_every = print_every
        self.writer = writer
        self.run_id_str = run_id_str
        self.experiment = experiment

    # verbose is for whether or not epochs are printed
    def train(self, verbose=False):
        best_model_wts = copy.deepcopy(self.model.state_dict())

        if self.device is None:
            self.model = self.model.cuda()  # move the model parameters to CPU/GPU 
        else:
            self.model = self.model.to(device=self.device)
        
        epochs_traning = np.zeros((self.epochs, 2))
        epochs_validation = np.zeros((self.epochs, 2))
        loss_all = np.zeros((len(self.loader_for_train)*self.epochs, 2))
        self.best_model = self.model
        best_acc = 0
        
        if verbose:
            print(self.epochs)
            print('Start training ... ')
            print(self.optimizer)
            print('--'  * 20)

        iter = 0
        val_acc_prev = 0
        num_epochs_low_acc = 0
        for e in range(self.epochs):
            running_loss = 0.0
            running_corrects = 0
            total_traning_size = 0

            t0 = time()
            for t, (x, y, n) in enumerate(self.loader_for_train):

                #if iter == 0:
                #    print("Add model to summary")
                    #writer.add_graph(model, x, verbose=False)

                self.model.train()  # put model to training mode
                if self.device is None:
                    x = x.to(self.x_dtype).cuda() 
                    y = y.to(self.y_dtype).cuda()
                else:
                    x = x.to(device=self.device, dtype=self.x_dtype) 
                    y = y.to(device=self.device, dtype=self.y_dtype)

                scores = self.model(x)
                loss = self.compute_loss(scores, y)

                if self.writer is not None:
                    self.writer.add_scalars(self.run_id_str + 'iterations',{'loss':loss}, iter)

                loss_all[iter, 0] = iter
                loss_all[iter, 1] = loss
                iter += 1

                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                self.optimizer.zero_grad()

                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                self.optimizer.step()

                # statistics
                running_loss += loss.item() * x.size(0)
                running_corrects += self.compute_running_accuracy(scores, y, x)
                total_traning_size += x.size(0)

                if t>0 and t % self.print_every == 0:
                    print('    Iterations %d, loss = %.4f' % (t, loss.item()))

                #progress_bar(t, len(loader_for_train), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (running_loss/(t+1), 100.*running_corrects/total_traning_size, running_corrects, total_traning_size))

            t1 = time()

            training_loss = running_loss / total_traning_size
            training_acc = running_corrects.double() / total_traning_size
            epochs_traning[e, 0] = training_loss
            epochs_traning[e, 1] = training_acc

            # validation
            self.model.eval() # model must be set to eval mode
            t0_val = time()
            val_acc, val_loss = self.check_validation_test_accuracy(self.loader_for_val, self.model)
            t1_val = time()
            epochs_validation[e, 0] = val_loss
            epochs_validation[e, 1] = val_acc
            if verbose:
                print('\033[1m' + 'Epochs %d/%d, takes %f/%f seconds for training/validation --- training accuracy = %.4f, validation accuracy = %.4f --- training loss = %.4f, validation loss = %.4f' % (e, self.epochs, t1-t0, t1_val-t0_val, training_acc, val_acc, training_loss, val_loss) + '\033[0m')
            #logger.debug('\033[1m' + 'Epochs %d/%d, takes %f seconds --- training accuracy = %.4f, validation accuracy = %.4f --- training loss = %.4f, validation loss = %.4f' % (e, epochs, t1-t0, training_acc, val_acc, training_loss, val_loss) + '\033[0m')
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                num_epochs_low_acc = 0 # once a better solution is found, reset early stopping
            else:
                num_epochs_low_acc += 1

            if self.writer is not None:
                self.writer.add_scalars(self.run_id_str + 'epochs_loss',{'training loss':training_loss, 'validation loss':val_loss}, e)
                self.writer.add_scalars(self.run_id_str + 'epochs_accuracy',{'training accuracy':training_acc, 'validation accuracy':val_acc}, e)

            if self.experiment:
                ### Save Metrics ###
                self.experiment.save_history('train', training_loss, training_acc)
                self.experiment.save_history('val', val_loss, val_acc)
                
                ### Checkpoint ###    
                self.experiment.save_weights(self.model, training_loss, val_loss, training_acc, val_acc)
                self.experiment.save_optimizer(self.optimizer, val_loss)
                
                ### Plot Online ###
                self.experiment.update_viz_loss_plot()
                self.experiment.update_viz_acc_plot()
                self.experiment.update_viz_summary_plot()

                self.experiment.epoch += 1

            if self.scheduler is not None:
                self.scheduler.step(val_acc)

            if self.exit_training(training_loss, training_acc, val_loss, val_acc):
                break
            
            if val_acc < val_acc_prev/2:
                print("validation accuracy goes way down ... ")
                break
                
            val_acc_prev = val_acc

            if num_epochs_low_acc>self.early_stopping_thres:
                print("Early stopping triggered : num_epochs_low_acc>self.early_stopping_thres ... %d > %d" % (num_epochs_low_acc, self.early_stopping_thres))
                break
            
        if self.writer is not None:
            self.writer.export_scalars_to_json("./all_scalars.json")

        self.model.load_state_dict(best_model_wts)
        return epochs_traning, epochs_validation, self.model, loss_all

    def check_validation_test_accuracy(self, loader, model):
        num_correct = 0
        num_samples = 0
        running_loss = 0.0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y, n in loader:
                if self.device is None:
                    x = x.to(self.x_dtype).cuda() 
                    y = y.to(self.y_dtype).cuda()
                else:
                    x = x.to(device=self.device, dtype=self.x_dtype) 
                    y = y.to(device=self.device, dtype=self.y_dtype)
    
                scores = model(x)
                loss = self.compute_loss(scores, y)
                num_correct += self.compute_running_accuracy(scores, y, x)
                num_samples += x.shape[0]
                running_loss += loss.item() * x.shape[0]
            acc = float(num_correct) / num_samples
            loss = running_loss/ num_samples
    
        return acc, loss

    def compute_running_accuracy(self, scores, y, x):
        r"""
        Compute dice score for accuracy
        """
        _, preds = torch.max(scores, 1)
        return torch.sum(preds == y.data)

    def compute_loss(self, scores, y):
        return self.criterion(scores, y)

    def exit_training(self, training_loss, training_acc, val_loss, val_acc):
        if training_acc>0.995:
            print("Training accuracy reaches 99.5%")
            return True
        return False

'''
Two Class segmentation 
'''
class GadgetronTrainerTwoClassSegmentation(GadgetronTrainer):
    r"""
    The trainer class for segmeantion, two class
    """
    def __init__(self, model, optimizer, criterion, loader_for_train, loader_for_val, scheduler=None, epochs=10, device=torch.device('cpu'), x_dtype=torch.float32, y_dtype=torch.uint8,  early_stopping_thres=10, print_every=100, writer = None, run_id_str="training/", experiment=None):
        super().__init__(model, optimizer, criterion, loader_for_train, loader_for_val, scheduler, epochs, device, x_dtype, y_dtype, early_stopping_thres, print_every, writer, run_id_str, experiment)

    def compute_running_accuracy(self, scores, y, x, p_thres=0.6):
        probs = torch.sigmoid(scores)
        y_pred = (probs > p_thres).float()

        accu = 0
        for n in range(y.shape[0]):
            accu += dice_coeff(y_pred[n,:], y[n,:])
        return accu

    def compute_loss(self, scores, y):
        probs = torch.sigmoid(scores)
        probs_flat = probs.view(-1)
        y_flat = y.view(-1)
        return self.criterion(probs_flat, y_flat.float())

    def exit_training(self, training_loss, training_acc, val_loss, val_acc):
        r"""
        For segmenation, do not stop the training...
        """
        return False

'''
Two Class segmentation , for perfusion AIF
'''
class GadgetronTrainerTwoClassSegmentation_PerfAIF(GadgetronTrainer):
    r"""
    The trainer class for segmeantion, two class
    """
    def __init__(self, model, optimizer, criterion, loader_for_train, loader_for_val, scheduler=None, epochs=10, device=torch.device('cpu'), x_dtype=torch.float32, y_dtype=torch.uint8,  early_stopping_thres=10, print_every=100, writer = None, run_id_str="training/", experiment=None):
        super().__init__(model, optimizer, criterion, loader_for_train, loader_for_val, scheduler, epochs, device, x_dtype, y_dtype, early_stopping_thres, print_every, writer, run_id_str, experiment)
        
    def compute_running_accuracy(self, scores, y, x):
        probs = torch.sigmoid(scores)
        N = probs.shape[0]
        
        y_pred = torch.zeros_like(probs)
        for i in range(N):
            y_pred[i,0,:,:] = adaptive_thresh(probs[i,0,:,:], self.device)
        #y_pred = (probs > torch.max(probs) * p_thresh).float()  # <-- This is wrong

        accu = 0
        for n in range(y.shape[0]):
            accu += dice_coeff(y_pred[n,:], y[n,:])
        return accu

    def compute_loss(self, scores, y):
        probs = torch.sigmoid(scores)
        probs_flat = probs.view(-1)
        y_flat = y.view(-1)
        return self.criterion(probs_flat, y_flat.float())

    def exit_training(self, training_loss, training_acc, val_loss, val_acc):
        r"""
        For segmenation, do not stop the training...
        """
        return False
    
    def compute_accuracy(self, loader, model, dice_thresh, dist_thresh):
        # Returns list consisting of tuples (name, Dice score, centroid distance)
        
        accus = []      
        model.eval()
        
        with torch.no_grad():
            for x, y, names in loader:
                if self.device is None:
                    x = x.to(self.x_dtype).cuda() 
                    y = y.to(self.y_dtype).cuda()
                else:
                    x = x.to(device=self.device, dtype=self.x_dtype) 
                    y = y.to(device=self.device, dtype=self.y_dtype)

                scores = model(x)
                batch_size, _, _ , _= x.shape
                
                for ii in range(batch_size):
                    name = names[ii]
                    img = scores[ii,0,:]
                    mask = y[ii,0,:]
                    aif_moco_echo1 = x[ii,:]
                    
                    probs = torch.sigmoid(img)
                    mask_pred = adaptive_thresh(probs, self.device)
                    
                    accu = dice_coeff(mask_pred, mask)
                    _, _, dist = centroid_diff(mask_pred, mask)
                    accus.append((name, accu, dist))
                    
                    #if(dist > dist_thresh):
                    #    scipy.io.savemat('NN_Output/' + name, \
                    #                     {'Prediction': mask_pred.cpu().numpy(), 'Probabilities': probs.cpu().numpy()})
        return accus
    
    def show_failed_dice_cases(self, loader, model, thresh):
        model.eval()  # set model to evaluation mode
        
        fig1 = plt.figure()
        ax0 = fig1.add_subplot(1,5,1)
        ax1 = fig1.add_subplot(1,5,2)
        ax2 = fig1.add_subplot(1,5,3)
        ax3 = fig1.add_subplot(1,5,4)
        ax4 = fig1.add_subplot(1,5,5)   
        ax5 = fig1.add_subplot(2,5,6)
        ax6 = fig1.add_subplot(2,5,7)
        ax7 = fig1.add_subplot(2,5,8)
        ax8 = fig1.add_subplot(2,5,9)
        ax9 = fig1.add_subplot(2,5,10)  
        
        fig2 = plt.figure()
        bx0 = fig2.add_subplot(1,5,1)
        bx1 = fig2.add_subplot(1,5,2)
        bx2 = fig2.add_subplot(1,5,3)
        bx3 = fig2.add_subplot(1,5,4)
        bx4 = fig2.add_subplot(1,5,5)
        
        with torch.no_grad():
            for x, y, names in loader:
                if self.device is None:
                    x = x.to(self.x_dtype).cuda() 
                    y = y.to(self.y_dtype).cuda()
                else:
                    x = x.to(device=self.device, dtype=self.x_dtype) 
                    y = y.to(device=self.device, dtype=self.y_dtype)

                scores = model(x)
                batch_size, _, _ , _= x.shape
                for ii in range(batch_size):
                    name = names[ii]
                    img = scores[ii,0,:]
                    mask = y[ii,0,:]
                    aif_moco_echo1 = x[ii,:]
                    
                    probs = torch.sigmoid(img)
                    mask_pred = adaptive_thresh(probs, self.device)
                    
                    accu = dice_coeff(mask_pred, mask)
                    if(accu < thresh):                        
                        print("Accuracy: %f" % accu)
                        print(name)
                        
                        ax0.imshow(x[ii,0,:,:])
                        ax1.imshow(x[ii,8,:,:])
                        ax2.imshow(x[ii,16,:,:])
                        ax3.imshow(x[ii,24,:,:])
                        ax4.imshow(x[ii,32,:,:])
                        ax5.imshow(x[ii,40,:,:])
                        ax6.imshow(x[ii,48,:,:])
                        ax7.imshow(x[ii,56,:,:])
                        ax8.imshow(x[ii,64,:,:])
                        ax9.imshow(x[ii,72,:,:])
                        
                        bx0.imshow(probs)
                        bx1.imshow(mask_pred)
                        bx2.imshow(mask)
                        
                        display(fig1)
                        display(fig2)
                        input("Press any key to continue:")
                        plt.close()
                        clear_output()
                        

    def show_failed_centroid_cases(self, loader, model, thresh):
        model.eval()  # set model to evaluation mode
        
        fig1 = plt.figure()
        ax0 = fig1.add_subplot(1,5,1)
        ax1 = fig1.add_subplot(1,5,2)
        ax2 = fig1.add_subplot(1,5,3)
        ax3 = fig1.add_subplot(1,5,4)
        ax4 = fig1.add_subplot(1,5,5)   
        ax5 = fig1.add_subplot(2,5,6)
        ax6 = fig1.add_subplot(2,5,7)
        ax7 = fig1.add_subplot(2,5,8)
        ax8 = fig1.add_subplot(2,5,9)
        ax9 = fig1.add_subplot(2,5,10)   
        
        fig2 = plt.figure()
        bx0 = fig2.add_subplot(1,5,1)
        bx1 = fig2.add_subplot(1,5,2)
        bx2 = fig2.add_subplot(1,5,3)
        bx3 = fig2.add_subplot(1,5,4)
        bx4 = fig2.add_subplot(1,5,5)
        
        with torch.no_grad():
            for x, y, names in loader:
                if self.device is None:
                    x = x.to(self.x_dtype).cuda() 
                    y = y.to(self.y_dtype).cuda()
                else:
                    x = x.to(device=self.device, dtype=self.x_dtype) 
                    y = y.to(device=self.device, dtype=self.y_dtype)

                scores = model(x)
                batch_size, _, _ , _= x.shape
                for ii in range(batch_size):
                    name = names[ii]
                    img = scores[ii,0,:]
                    mask = y[ii,0,:]
                    aif_moco_echo1 = x[ii,:]
                    
                    probs = torch.sigmoid(img)
                    mask_pred = adaptive_thresh(probs, self.device)
  
                    pred_centroid, mask_centroid, accu = centroid_diff(mask_pred, mask)
                    if(accu > thresh):
                        print("Distance: %f" % accu)
                        print(name)
                        
                        ax0.imshow(x[ii,0,:,:])
                        ax1.imshow(x[ii,8,:,:])
                        ax2.imshow(x[ii,16,:,:])
                        ax3.imshow(x[ii,24,:,:])
                        ax4.imshow(x[ii,32,:,:])
                        ax5.imshow(x[ii,40,:,:])
                        ax6.imshow(x[ii,48,:,:])
                        ax7.imshow(x[ii,56,:,:])
                        ax8.imshow(x[ii,64,:,:])
                        ax9.imshow(x[ii,72,:,:])
                        
                        bx0.imshow(probs)
                        bx1.imshow(mask_pred)
                        bx2.imshow(mask)
                        bx2.plot(pred_centroid[0],pred_centroid[1],'r+') # Prediction in red
                        bx2.plot(mask_centroid[0],mask_centroid[1],'g+') # Ground truth in green
                        
                        display(fig1)
                        display(fig2)
                        input("Press any key to continue:")
                        bx2.clear()
                        plt.close()
                        clear_output()

def train_test(model, 
               optimizer, 
               scheduler, 
               criterion, 
               loader_for_train, 
               loader_for_val, 
               epochs=10, 
               device=torch.device('cpu'), 
               dtype=torch.float32, 
               print_every=100,
               writer = None, 
               compute_acc = True,
               run_id_str="training/"):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """

    best_model_wts = copy.deepcopy(model.state_dict())

    if device is None:
        model = model.cuda()  # move the model parameters to CPU/GPU 
    else:
        model = model.to(device=device)
    print(epochs)
    epochs_traning = np.zeros((epochs, 2))
    epochs_validation = np.zeros((epochs, 2))
    loss_all = np.zeros((len(loader_for_train)*epochs, 2))
    best_model = model
    best_acc = 0

    ''' writer.add_scalar('Summary/epochs', epochs)
    if device == torch.device('cpu'):
        writer.add_scalar('Summary/cpu', 1)
    if device == torch.device('cuda'):
        writer.add_scalar('Summary/cuda', 1) '''

    print('Start training ... ')
    print('--'  * 20)

    iter = 0
    for e in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        total_traning_size = 0

        t0 = time()
        for t, (x, y) in enumerate(loader_for_train):

            #if iter == 0:
            #    print("Add model to summary")
                #writer.add_graph(model, x, verbose=False)

            model.train()  # put model to training mode
            if device is None:
                x = x.to(dtype).cuda() 
                y = y.to(torch.long).cuda()
            else:
                x = x.to(device=device, dtype=dtype) 
                y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            if compute_acc:
                _, preds = torch.max(scores, 1)
            loss = criterion(scores, y)

            if writer is not None:
                writer.add_scalars(run_id_str + 'iterations',{'loss':loss}, iter)

            loss_all[iter, 0] = iter
            loss_all[iter, 1] = loss
            iter += 1

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            # statistics
            running_loss += loss.item() * x.size(0)
            if compute_acc:
                running_corrects += torch.sum(preds == y.data)
            total_traning_size += x.size(0)

            if t>0 and t % print_every == 0:
                print('    Iterations %d, loss = %.4f' % (t, loss.item()))

            #progress_bar(t, len(loader_for_train), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (running_loss/(t+1), 100.*running_corrects/total_traning_size, running_corrects, total_traning_size))

        t1 = time()

        training_loss = running_loss / total_traning_size
        training_acc = running_corrects.double() / total_traning_size
        epochs_traning[e, 0] = training_loss
        epochs_traning[e, 1] = training_acc

        # validation
        model.eval() # model must be set to eval mode
        t0_val = time()
        val_acc, val_loss = check_accuracy_test(loader_for_val, model, criterion, device, dtype)
        t1_val = time()
        epochs_validation[e, 0] = val_loss
        epochs_validation[e, 1] = val_acc
        print('\033[1m' + 'Epochs %d/%d, takes %f/%f seconds for training/validation --- training accuracy = %.4f, validation accuracy = %.4f --- training loss = %.4f, validation loss = %.4f' % (e, epochs, t1-t0, t1_val-t0_val, training_acc, val_acc, training_loss, val_loss) + '\033[0m')
        #logger.debug('\033[1m' + 'Epochs %d/%d, takes %f seconds --- training accuracy = %.4f, validation accuracy = %.4f --- training loss = %.4f, validation loss = %.4f' % (e, epochs, t1-t0, training_acc, val_acc, training_loss, val_loss) + '\033[0m')
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if writer is not None:
            writer.add_scalars(run_id_str + 'epochs_loss',{'training loss':training_loss, 'validation loss':val_loss}, e)
            writer.add_scalars(run_id_str + 'epochs_accuracy',{'training accuracy':training_acc, 'validation accuracy':val_acc}, e)

        if scheduler is not None:
            scheduler.step(val_acc)

        if training_acc>0.995:
            # print("Training accuracy reaches 99.5%")
            print("Training accuracy reaches 99.5%")
            break

    if writer is not None:
        writer.export_scalars_to_json("./all_scalars.json")

    model.load_state_dict(best_model_wts)
    return epochs_traning, epochs_validation, model, loss_all
