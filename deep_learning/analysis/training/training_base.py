
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm_notebook
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
The cost and accuray functions are for the classification problem
'''
class GadgetronTrainer(object):
    def __init__(self, model, optimizer, criterion, 
                 loader_for_train, loader_for_val, 
                 scheduler=None, epochs=10, 
                 device=torch.device('cpu'), 
                 x_dtype=torch.float32, y_dtype=torch.long, 
                 early_stopping_thres=10, 
                 print_every=100, 
                 small_data_mode = False,
                 writer = None, 
                 model_folder="training/"):
        r"""
        Define the trainer class for Gadgetron NN
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
        self.model_folder = model_folder
        self.small_data_mode = False
        
    # verbose is for whether or not epochs are printed
    def train(self, verbose=False, epoch_to_load=-1, save_model_epoch=False):
        best_model_wts = copy.deepcopy(self.model.state_dict())

        if self.device is None:
            self.model = self.model.cuda()  # move the model parameters to CPU/GPU 
        else:
            self.model = self.model.to(device=self.device)

        epochs_traning = np.zeros((self.epochs, 2))
        epochs_validation = np.zeros((self.epochs, 2))
        epochs_accuracy_class = []
        loss_all = np.zeros((len(self.loader_for_train)*self.epochs, 2))
        self.best_model = self.model
        best_acc = 0
        e = 0
        step = 0
        
        train_size = len(self.loader_for_train)
        batch_size = self.loader_for_train.batch_size
        
        # -------------------------------------------------------------------
        # load models if needed
        model_path = os.path.join(self.model_folder,  'model_epoch_')
        if(epoch_to_load>=0):
            model_path +=  str(epoch_to_load) + '.pbt'
            print("Check model file : ", model_path)
            if os.path.isfile(model_path):
                state = torch.load(str(model_path))
                e = state['epoch']
                # step = state['step']
                step = e * train_size
                self.model.load_state_dict(state['model'])
                print('Restored model, epoch {}, step {:,}'.format(e, step))
    
        # -------------------------------------------------------------------
        # save model function
        save = lambda ep: torch.save({
            'model': self.model.state_dict(),
            'epoch': ep,
            'step': step,
        }, str(model_path + str(ep) + '.pbt'))
    
        # -------------------------------------------------------------------
    
        if verbose:
            print(self.epochs)
            print('Start training ... ')
            print(self.optimizer)
            print('--'  * 20)

        val_acc_prev = 0
        num_epochs_low_acc = 0
        
        # ======================================================================================================================= #
        # run epochs
        for e in range(e, self.epochs):
           
            running_loss = 0.0
            running_corrects = 0
            total_traning_size = 0

            random.seed()
            tq = tqdm_notebook(total=(len(self.loader_for_train) * batch_size), file=sys.stdout)
            tq.set_description('Epoch {}, total {}'.format(e, self.epochs))
        
            try:
                t0 = time()

                # ================================================================== #
                # run training examples
                for t, (x, y, n) in enumerate(self.loader_for_train):

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
                        self.writer.add_scalars(self.model_folder + '/iterations',{'loss':loss}, step)

                    loss_all[step, 0] = step
                    loss_all[step, 1] = loss
                    step += 1

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * x.size(0)
                    curr_correct, curr_correct_class, curr_dice_all = self.compute_running_accuracy(scores, y, x)
                    running_corrects += curr_correct
                    total_traning_size += x.size(0)

                    tq.update(batch_size)
                    tq.set_postfix(loss='{:.5f}'.format(loss.item()))

                    if t>0 and t % self.print_every == 0:
                        print('    Iterations %d, loss = %.4f' % (t, loss.item()))

                t1 = time()

                # ================================================================== #

                training_loss = running_loss / total_traning_size
                training_acc = running_corrects / total_traning_size
                epochs_traning[e, 0] = training_loss
                epochs_traning[e, 1] = training_acc           

                # -------------------------------------------------------------------
                # validation
                self.model.eval() # model must be set to eval mode
                t0_val = time()
                val_acc, val_loss, val_acc_class = self.check_validation_test_accuracy(self.loader_for_val, self.model)
                t1_val = time()
                epochs_validation[e, 0] = val_loss
                epochs_validation[e, 1] = val_acc
                epochs_accuracy_class.append(val_acc_class)
                
                # -------------------------------------------------------------------
                # string outputs
                str_after_val = '    %.2f/%.2f seconds for Training/Validation --- Tra acc = %.4f, Val acc = %.4f --- Tra loss = %.4f, Val loss = %.4f, --- class acc = ' % (t1-t0, t1_val-t0_val, training_acc, val_acc, training_loss, val_loss)

                for c in range(val_acc_class.shape[0]-1):
                    str_after_val += str(val_acc_class[c]) + ', '
                str_after_val += str(val_acc_class[val_acc_class.shape[0]-1])
                
                tq.set_postfix_str(str_after_val)
                tq.close() 

                print(str_after_val)
                
                # -------------------------------------------------------------------
                # save model for this epoch
                if(save_model_epoch):
                    save(e)

                # -------------------------------------------------------------------
                # save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    num_epochs_low_acc = 0 # once a better solution is found, reset early stopping
                else:
                    num_epochs_low_acc += 1

                # -------------------------------------------------------------------
                # tensorboardx record
                if self.writer is not None:
                    self.writer.add_scalars(self.model_folder + 'epochs_loss',{'training loss':training_loss, 'validation loss':val_loss}, e)
                    self.writer.add_scalars(self.model_folder + 'epochs_accuracy',{'training accuracy':training_acc, 'validation accuracy':val_acc}, e)

                # -------------------------------------------------------------------
                if self.scheduler is not None:
                    self.scheduler.step(val_acc)

                # -------------------------------------------------------------------                
                if self.exit_training(training_loss, training_acc, val_loss, val_acc):
                    break

                # -------------------------------------------------------------------    
                if(self.small_data_mode == False):
                    if val_acc < val_acc_prev/10:
                        print("validation accuracy goes way down ... ")
                        break

                val_acc_prev = val_acc

                if num_epochs_low_acc>self.early_stopping_thres:
                    print("Early stopping triggered : num_epochs_low_acc>self.early_stopping_thres ... %d > %d" % (num_epochs_low_acc, self.early_stopping_thres))
                    break
                    
            except KeyboardInterrupt:

                tq.close()
                print('Ctrl+C, saving snapshot')
                save(e)
                print('done.')

                self.model.load_state_dict(best_model_wts)
                return epochs_traning, epochs_validation, self.model, loss_all, epochs_accuracy_class

        # ======================================================================================================================= #

        if self.writer is not None:
            self.writer.export_scalars_to_json("./all_scalars.json")

        self.model.load_state_dict(best_model_wts)
        return epochs_traning, epochs_validation, self.model, loss_all, epochs_accuracy_class

    def check_validation_test_accuracy(self, loader, model):
        num_correct = 0
        num_samples = 0
        running_loss = 0.0
        correct_class = None
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
                curr_correct, curr_correct_class, curr_dice_all = self.compute_running_accuracy(scores, y, x)
                num_correct += curr_correct
                num_samples += x.shape[0]
                running_loss += loss.item() * x.shape[0]

                if(correct_class is None):
                    correct_class = curr_correct_class
                else:
                    correct_class += curr_correct_class

            acc = float(num_correct) / num_samples
            loss = running_loss/ num_samples
            acc_class= correct_class/num_samples

        return acc, loss, acc_class

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
