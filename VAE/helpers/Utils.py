
import wandb
import os 

def save_images_wandb(wandb):
    dir_path = "/app/ai_experiments/Lockout/tmp/"
    for idx, fig_path in enumerate(os.listdir(dir_path)):
        if fig_path.endswith(".png"):
            wandb.log({f"{fig_path.replace('.png', '')}": wandb.Image(dir_path+fig_path)})