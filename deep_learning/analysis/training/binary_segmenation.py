
from .training_base import *

'''
Two Class segmentation 
'''
class GadgetronTwoClassSeg(GadgetronTrainer):
    r"""
    The trainer class for segmeantion, two class
    """
    def __init__(self, model, optimizer, criterion, loader_for_train, loader_for_val, scheduler=None, epochs=10, device=torch.device('cpu'), x_dtype=torch.float32, y_dtype=torch.uint8,  early_stopping_thres=10, print_every=100, small_data_mode = False, writer = None, model_folder="training/"):
        super().__init__(model, optimizer, criterion, loader_for_train, loader_for_val, scheduler, epochs, device, x_dtype, y_dtype, early_stopping_thres, print_every, small_data_mode, writer, model_folder)

    def compute_running_accuracy(self, scores, y, x, p_thres=0.6):
        probs = torch.sigmoid(scores)
        y_pred = (probs > p_thres).float()

        dice_all = np.zeros((y.shape[0], 1))
        
        accu = 0
        accu_class = np.zeros(1)
        for n in range(y.shape[0]):
            d = dice_coeff(y_pred[n,:], y[n,:])
            accu += d
            dice_all[n, 0] = d
            
        accu_class[0] = accu
        return accu, accu_class, dice_all

    def compute_loss(self, scores, y):
        probs = torch.sigmoid(scores)
        probs_flat = probs.view(-1)
        y_flat = y.view(-1)
        return self.criterion(probs_flat, y_flat)

    def exit_training(self, training_loss, training_acc, val_loss, val_acc):
        r"""
        For segmenation, do not stop the training...
        """
        return False

'''
Two Class segmentation , for perfusion AIF
'''
class GadgetronTwoClassSeg_PerfAIF(GadgetronTrainer):
    r"""
    The trainer class for segmeantion, two class
    """
    def __init__(self, model, optimizer, criterion, loader_for_train, loader_for_val, p_thres=0.5, scheduler=None, epochs=10, device=torch.device('cpu'), x_dtype=torch.float32, y_dtype=torch.uint8,  early_stopping_thres=10, print_every=100, small_data_mode=False, writer = None, model_folder="training/"):
        super().__init__(model, optimizer, criterion, loader_for_train, loader_for_val, scheduler, epochs, device, x_dtype, y_dtype, early_stopping_thres, print_every, small_data_mode, writer, model_folder)

        self.p_thres = p_thres

    def compute_running_accuracy(self, scores, y, x):
        probs = torch.sigmoid(scores)
        N = probs.shape[0]

        dice_all = np.zeros((y.shape[0], 1))
        
        y_pred = torch.zeros_like(probs)
        for i in range(N):
            y_pred[i,0,:,:] = adaptive_thresh(probs[i,0,:,:], self.device, self.p_thres)

        accu = 0
        accu_class = np.zeros(1)
        for n in range(y.shape[0]):
            d = dice_coeff(y_pred[n,0,:,:], y[n,0,:,:])
            accu += d
            dice_all[n, 0] = d

        accu_class[0] = accu
        return accu, accu_class, dice_all

    def compute_loss(self, scores, y):
        return self.criterion(scores, y)

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
                    mask_pred = adaptive_thresh(probs, self.device, self.p_thres)

                    accu = dice_coeff(mask_pred.cpu(), mask.cpu())
                    _, _, dist = centroid_diff(mask_pred.cpu(), mask.cpu())
                    accus.append((name, accu, dist))
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
                    mask_pred = adaptive_thresh(probs, self.device, self.p_thres)

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
                    mask_pred = adaptive_thresh(probs, self.device, self.p_thres)
  
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

# for cine segmenation
class GadgetronTwoClassSeg_Cine(GadgetronTrainer):
    r"""
    The trainer class for segmeantion, two class
    """
    def __init__(self, model, optimizer, criterion, loader_for_train, loader_for_val, p_thres=0.5, min_prob_allowed=0.4, scheduler=None, epochs=10, device=torch.device('cpu'), x_dtype=torch.float32, y_dtype=torch.uint8,  early_stopping_thres=10, print_every=100, small_data_mode=False, writer = None, model_folder="training/"):
        super().__init__(model, optimizer, criterion, loader_for_train, loader_for_val, scheduler, epochs, device, x_dtype, y_dtype, early_stopping_thres, print_every, small_data_mode, writer, model_folder)

        self.p_thres = p_thres
        self.min_prob_allowed = min_prob_allowed
        
    def compute_running_accuracy(self, scores, y, x):
        probs = torch.sigmoid(scores)
        N = probs.shape[0]

        dice_all = np.zeros((y.shape[0], 1))
        
        y_pred = torch.zeros_like(probs)
        for i in range(N):
            max_prob = torch.max(probs[i,0,:,:])
            if(max_prob>self.min_prob_allowed):
                y_pred[i,0,:,:] = adaptive_thresh(probs[i,0,:,:], self.device, self.p_thres)

        accu = 0
        accu_class = np.zeros(1)
        for n in range(y.shape[0]):
            max_y_pred = torch.max(y_pred[n,0,:,:])
            max_y = torch.max(y[n,0,:,:])
            
            if(max_y_pred<1e-2 and max_y<1e-2):
                d = 1.0
            else:
                d = dice_coeff(y_pred[n,0,:,:], y[n,0,:,:])
            accu += d
            dice_all[n, 0] = d

        accu_class[0] = accu
        return accu, accu_class, dice_all

    def compute_loss(self, scores, y):
        return self.criterion(scores, y)

    def exit_training(self, training_loss, training_acc, val_loss, val_acc):
        r"""
        For segmenation, do not stop the training...
        """
        return False                        