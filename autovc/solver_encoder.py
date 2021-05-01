from autovc.model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime

import os


class Solver(object):

    def __init__(self, vcc_loader, config, device):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = device
        self.log_step = config.log_step
        self.save_freq = config.save_freq
        self.checkpoint_dir= config.checkpoint_dir

        if not os.path.exists (self.checkpoint_dir): #create checkpoint dir if it does not exist
            os.makedirs(self.checkpoint_dir)

        # Build the model and tensorboard.
        self.build_model()
        self.model_path = config.model_path
        
        self.start_step = 0
        
        if self.model_path is not None:
            g_checkpoint = torch.load(self.model_path, map_location=self.device) 
            self.G.load_state_dict(g_checkpoint['model'])
            
            if "optimizer" in g_checkpoint.keys():
                self.g_optimizer.load_state_dict(g_checkpoint["optimizer"])
            else:
                print("WARNING: didn't load optimizer state!!!")
                
            if "steps" in g_checkpoint.keys():
                self.start_step = g_checkpoint["steps"]

            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        
        self.G.to(self.device)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
        
        loss = {}
        for key in keys:
            loss[key] = [] #create loss histories
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
            g_loss_id = F.mse_loss(x_real, torch.reshape(x_identic, (x_real.shape)))   # TODO: added: reshape tensors to fix user warning
            g_loss_id_psnt = F.mse_loss(x_real, torch.reshape(x_identic_psnt, (x_real.shape)))   # TODO: added: reshape tensors to fix user warning
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss['G/loss_id'].append( (i, g_loss_id.item()) )
            loss['G/loss_id_psnt'].append( (i,g_loss_id_psnt.item()) )
            loss['G/loss_cd'].append( (i,g_loss_cd.item()) )

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, self.start_step+i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag][-1][1])
                print(log)
                
            if (i+1) % self.save_freq == 0:
                torch.save({
                            "model" : self.G.state_dict(), 
                            'optimizer': self.g_optimizer.state_dict(),
                            "steps" : self.start_step+i,
                            "loss" : loss, 
                            }, os.path.join(self.checkpoint_dir, "autovc_{}.ckpt".format(self.start_step+i+1)))
                

    
    

    