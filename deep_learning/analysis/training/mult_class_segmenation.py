
from .training_base import *

'''
Multi-class segmentation 
'''
class GadgetronMultiClassSeg(GadgetronTrainer):
    r"""
    The trainer class for segmeantion, multi-class
    """
    def __init__(self, model, optimizer, criterion, loader_for_train, loader_for_val, scheduler=None, epochs=10, device=torch.device('cpu'), x_dtype=torch.float32, y_dtype=torch.uint8,  early_stopping_thres=10, print_every=100, small_data_mode = False, writer = None, model_folder="training/"):
        super().__init__(model, optimizer, criterion, loader_for_train, loader_for_val, scheduler, epochs, device, x_dtype, y_dtype, early_stopping_thres, print_every, small_data_mode, writer, model_folder)

    def compute_running_accuracy(self, scores, y, x, p_thres=0.6):

        N = scores.shape[0]
        C = scores.shape[1]

        m = torch.nn.Softmax(dim=1)
        probs = m(scores)

        dice_all = np.zeros((N, C))
        
        # first class is background
        accu = 0
        accu_class = np.zeros(C-1)
        for c in range(1, C):
            curr_prob = probs[:,c,:,:]
            y_pred = (curr_prob > p_thres).float()

            y_target = (y == c).float()

            for n in range(y.shape[0]):
                curr_dice = dice_coeff(y_pred[n,:], y_target[n,:])
                accu += curr_dice
                accu_class[c-1] += curr_dice
                dice_all[n, c] = curr_dice
                
        return accu/(C-1), accu_class, dice_all

    def compute_loss(self, scores, y):
        if(len(y.shape)==4 and y.shape[1]==1):
            return self.criterion(scores, y.reshape((y.shape[0], y.shape[2], y.shape[3])))
        else:
            return self.criterion(scores, y)

    def exit_training(self, training_loss, training_acc, val_loss, val_acc):
        r"""
        For segmenation, do not stop the training...
        """
        return False

'''
Multi-class segmentation, for perfusion SAX images
Use the LV and myocardium dice score as the accuracy
'''
class GadgetronMultiClassSeg_Perf(GadgetronTrainer):
    r"""
    The trainer class for perfusion segmeantion
    """
    def __init__(self, model, optimizer, criterion, loader_for_train, loader_for_val, class_for_accu, p_thres, scheduler=None, epochs=10, device=torch.device('cpu'), x_dtype=torch.float32, y_dtype=torch.uint8,  early_stopping_thres=10, print_every=100, small_data_mode=False, writer = None, model_folder="training/"):
        super().__init__(model, optimizer, criterion, loader_for_train, loader_for_val, scheduler, epochs, device, x_dtype, y_dtype, early_stopping_thres, print_every, small_data_mode, writer, model_folder)

        # classes to compute accuracy
        self.class_for_accu = class_for_accu
        # for every class, the threshold for adpative thresholding
        self.p_thres = p_thres

    def compute_running_accuracy(self, scores, y, x):

        N = scores.shape[0]
        C = scores.shape[1]

        C_accu = len(self.class_for_accu)        
        if(C_accu==0):
            C_accu = C-1
            self.class_for_accu = [ac for ac in range(1,C)]

        dice_all = np.zeros((N, C_accu))
        
        m = torch.nn.Softmax(dim=1)
        probs = m(scores)

        y_pred = torch.zeros_like(probs)
        accu = 0
        accu_class = np.zeros(C_accu)
        for i in range(N):
            for ac in range(C_accu):

                c = self.class_for_accu[ac]
                if(c>=C):
                    break

                curr_prob = probs[i,c,:,:]
                y_pred[i,c,:,:] = adaptive_thresh(curr_prob, self.device, self.p_thres[ac])

                y_target = torch.squeeze(y[i,:,:])
                y_target = y_target.cpu().float()
                y_target = (y_target == c).float()

                curr_dice = dice_coeff(torch.squeeze(y_pred[i,c,:,:]).cpu().float(), y_target)
                accu += curr_dice
                accu_class[ac] += curr_dice

                dice_all[i, ac] = curr_dice
                
        return accu/C_accu, accu_class, dice_all

    def compute_loss(self, scores, y):
        if(len(y.shape)==4 and y.shape[1]==1):
            return self.criterion(scores, y.reshape((y.shape[0], y.shape[2], y.shape[3])))
        else:
            return self.criterion(scores, y)

    def exit_training(self, training_loss, training_acc, val_loss, val_acc):
        r"""
        For segmenation, do not stop the training...
        """
        return False
