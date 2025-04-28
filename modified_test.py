"""General-purpose test script for image-to-image translation.
"""

import os
from options.test_options   import TestOptions
from data                   import create_dataset
from models                 import create_model
from util.visualizer        import save_images
from util                   import html
from models.pix7mask_model  import Pix7MaskModel
from tqdm                   import tqdm

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    
    # hard-code some parameters for test
    opt.num_threads = 0            # test code only supports num_threads = 0
    opt.batch_size = 1             # test code only supports batch_size = 1
    opt.serial_batches = True      # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True             # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1            # no visdom display; the test code saves the results to a HTML file.
    
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)  
    
    # Create the Model
    model : Pix7MaskModel = create_model(opt)   # create a model given opt.model and other options
    model.setup(opt)                            # regular setup: load and print networks; create schedulers
    model.eval()                                # set the model on evaluation mode

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    #if opt.load_iter > 0:  # load_iter is 0 by default
    #    web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    
    print('creating web directory', web_dir)
    
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    for i, data in enumerate(tqdm(dataset)):
        
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        
        visuals  = model.get_current_visuals_test_time()  # get image results
        img_path = model.get_image_paths()                # get image paths
        
        #if i % 5 == 0:  # save images to an HTML file
        #    print('processing (%04d)-th image... %s' % (i, img_path))
        
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    
    webpage.save()  # save the HTML
