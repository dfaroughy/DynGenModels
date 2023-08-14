import torch

class FlowMatchPipeline:
    
    def __init__(self, model, scheduler):
        super().__init__()

    def __call__(self, batch_size: int = 1, num_inference_steps: int = 50):

        noise = torch.randn((batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size))

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, eta).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return image