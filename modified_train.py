"""General-purpose training script for image-to-image translation.

Heavily Modified Script for Training Pix7Mask Models (there is no easy & number to use)
"""
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import time

from options.train_options  import TrainOptions
from data                   import create_dataset
from models                 import create_model
from models.pix7mask_model  import Pix7MaskModel
from util.visualizer        import Visualizer
from copy                   import deepcopy
from util.dictonary_tracker import GenericDictTracker
from util.dictonary_tracker import AttachedTracker


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    ## Insanity check
    if (opt.model != "pix7mask"): raise Exception("No Peko !")

    print("\n# Argument Parsed \n")

    ## Create Training Dataseet
    training_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(training_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    
    ## Create Evaluation Dataset
    evl_opt = deepcopy(opt)
    evl_opt.phase      = "eval"
    evl_opt.batch_size = 1
    eval_dataset = create_dataset(evl_opt)  # create a dataset given opt.dataset_mode and other options
    eval_size    = len(eval_dataset)        # get the number of images in the dataset.
    print('The number of evaluation images = %d' % eval_size)

    ## Define the model
    model : Pix7MaskModel = create_model(opt) # create a model given opt.model and other options
    model.setup(opt) # regular setup: load and print networks; create schedulers

    ## Define visualiser
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    metric_trackers = AttachedTracker(opt)
    print("\n# Train Start \n")

    ## Training - Total Epochs
    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    
        epoch_start_time = time.time()  # timer for entire epoch        
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        # setup model for training - update learning rates in the beginning of every epoch.
        training_variables = GenericDictTracker()
        model.train()
        model.update_learning_rate()

        ## Training Loop
        for train_iter, data in enumerate(training_dataset):  # inner loop within one epoch
            iter_start_time = time.time()   # timer for computation per iteration

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, train_iter, losses, t_comp)
            training_variables.append(losses)
        
        print("# Training Values", training_variables.calculate_averages())
        
        ## Evaluation Loop 
        evaluation_variables = GenericDictTracker()
        model.eval()
        for eval_iters, data in enumerate(eval_dataset):  # inner loop within one epoch
            iter_start_time = time.time()   # timer for computation per iteration

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.test()           # run inference

            # display images on visdom and save images to a HTML file
            if total_iters % opt.display_freq == 0:   
                save_result = total_iters % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            for ss_index in range(0, model.get_current_batch_size()):
                eval_values = model.compute_single_batch_acc(ss_index)
                evaluation_variables.append(eval_values)
                visualizer.print_current_losses(epoch, eval_iters, eval_values, t_comp, prefix = "E")
        
        print("# Evaluation Values", evaluation_variables.calculate_averages())

        # cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:              
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
        
        # Always save latest model
        model.save_networks('latest')

        metric_trackers.append(training_variables, evaluation_variables)
        if metric_trackers.is_current_best:
            model.save_networks("best")
        
        metric_trackers.write()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    print("$$ Model Training is Completed $$")