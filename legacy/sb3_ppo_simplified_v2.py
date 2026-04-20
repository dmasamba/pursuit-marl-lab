import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from robosuite.wrappers import GymWrapper
import robosuite as suite

import os

from stable_baselines3.common.callbacks import BaseCallback
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

envName = "reachImgSimplified"
img_dim = 64
use_proprio_obs = False 
# Create environment instance
env = suite.make(
    # env_name="Reach",
    # camera_names="frontview",
    # robots="UR5e",
    env_name=envName,
    camera_names=["topdown"],
    robots="UR5ev2",
    has_offscreen_renderer=True,
    use_camera_obs=True,
    use_object_obs=False,  # Exclude object observations
    camera_heights=img_dim,
    camera_widths=img_dim,
    reward_shaping=True
)

# Wrap the environment
vec_env = GymWrapper(env)

###############################################################



class ImageLoggingCallback(BaseCallback):
    """
    A custom callback that logs or saves images / feature maps at intervals.
    """
    def __init__(
        self,
        log_dir: str = "./logs/tensorboard",
        log_freq: int = 10000,
        n_images: int = 1,
        verbose: int = 0,
    ):
        super(ImageLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.n_images = n_images

        # Create log dir if needed
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        """
        This method is called at each call to `env.step()` in the training loop.
        We only proceed if the current timestep is divisible by log_freq.
        """
        # self.n_calls = number of calls to the callback
        # self.model.num_timesteps = total timesteps so far

        if self.model.num_timesteps % self.log_freq == 0:
            if self.verbose > 0:
                print(f"Logging images at step {self.model.num_timesteps}")

            # 1) Sample a mini-batch of observations
            obs = self._sample_observations()

            # 2) Pass observations through the policy -> get CNN outputs
            self._log_cnn_features(obs)

        return True

    def _sample_observations(self):
        """
        Here, we sample some observations from the environment.
        The simplest approach is to call `env.reset()` or `env.step()`.
        You may want to keep a reference to the training environment in the callback.
        """
        # Make sure we have direct access to the env
        # If you are using a VecEnv, you can do something like:
        obs = self.training_env.reset()

        # If needed, do a few steps to get variety
        # for i in range(5):
        #     action = [self.training_env.action_space.sample()]
        #     obs, rewards, dones, info = self.training_env.step(action)

        # Return observation(s)
        return obs

    def _log_cnn_features(self, obs):
        # obs shape is (num_envs, observation_dim).
        # We'll just look at the first environment:
        obs = obs[0]  # shape = (observation_dim,)
        obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(self.model.device)

        # 1) Slice out the image part (12,288 dims)
        img_size = 3 * 64 * 64  # 12288
        image_obs = obs[:, :img_size]  # shape = [1, 12288]

        # 2) Reshape for CNN
        image_obs = image_obs.view(-1, 3, 64, 64)

        # 3) Forward pass through the CNN
        with torch.no_grad():
            cnn_module = self.model.policy.features_extractor.cnn
            cnn_output, feature_maps = cnn_module(image_obs)


        # Log each feature map block
        for i, f_map in enumerate(feature_maps):
            # f_map shape: (batch_size, channels, H, W)
            # Let's just take the first sample in the batch:
            # (channels, H, W)
            f_map_single = f_map[0]

            # Often, feature maps have many channels, so you can visualize, say, the first few
            # or make a grid of channels:
            grid_feat = make_grid(
                f_map_single.unsqueeze(1),  # shape => (channels, 1, H, W)
                nrow=8,  # how many images per row
                normalize=True,
                scale_each=True
            )
            self.writer.add_image(f"feature_map_{i}", grid_feat, global_step=self.model.num_timesteps)

        self.writer.flush()

    def _on_training_end(self) -> None:
        """
        Called at the end of training, close out the SummaryWriter.
        """
        self.writer.close()


############################################


class CNNFeatures(nn.Module):
    def __init__(self):
        super(CNNFeatures, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        feat1 = x.clone()
        x = self.conv2(x)
        x = self.relu2(x)
        feat2 = x.clone()
        x = self.conv3(x)
        x = self.relu3(x)
        feat3 = x.clone()
        x = self.flatten(x)
        return x, [feat1, feat2, feat3]

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=img_dim, use_proprio_obs=False):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # CNN to process the image
        self.cnn = CNNFeatures()
        self.use_proprio_obs = use_proprio_obs
        
        # Output dimension of the CNN
        with torch.no_grad():
            sample_image = torch.zeros(1, 3, img_dim, img_dim)
            cnn_output, _ = self.cnn(sample_image)
            cnn_output_dim = cnn_output.shape[1]
        
        if self.use_proprio_obs:
            self.proprio_dim = 41
        else:
            self.proprio_dim = 0
        
        # Total features dimension
        self._features_dim = cnn_output_dim + self.proprio_dim
        # print("Total obs",self._features_dim)
    
    def forward(self, observations):
        batch_size = observations.shape[0]
        
        # Known dimensions
        image_dim = img_dim * img_dim * 3  # for 64x64 12,288, for 128x128 196,608
        
        # Split observations
        camview_image = observations[:, :image_dim]
        robot0_proprio_state = observations[:, image_dim:]
        
        # print("camview_image", camview_image.shape)
        # print("robot0_proprio_state", robot0_proprio_state.shape)
        # exit()

        # Process the image
        # Reshape image to (batch_size, img_dim, img_dim, 3)
        camview_image = camview_image.view(batch_size, img_dim, img_dim, 3)
        # Convert image to channels-first format (batch_size, 3, img_dim, img_dim)
        camview_image = camview_image.permute(0, 3, 1, 2)
        # Normalize image
        camview_image = camview_image / 255.0
        # Pass through CNN
        cnn_output, _ = self.cnn(camview_image)
        
        # Concatenate CNN output and proprioceptive state
        if self.use_proprio_obs:
            features = torch.cat([cnn_output, robot0_proprio_state], dim=1)
        else:
            features = cnn_output
        return features

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs={'features_dim': img_dim},
            **kwargs
        )




device ='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# Instantiate the PPO model with the custom policy
model = PPO(
    policy=CustomActorCriticPolicy,
    env=vec_env,
    verbose=1,
    tensorboard_log="./runs/ImgPPO_"+envName+"_sb3_simplifiedv3/",
    device = device
)

# Train the model
# model.learn(total_timesteps=2000000)

# # model.save("./runs/ImgPPO_"+envName+"_sb3/"+envName+"_Img_ppo_reach_simplified_1")

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# model.save(f"./runs/ImgPPO_{envName}_sb3/{envName}_Img_ppo_reach_simplified_1_{timestamp}")


# Instantiate your callback
image_logging_callback = ImageLoggingCallback(
    log_dir="./runs/ImgPPO_"+envName+"_sb3_simplifiedv3/tf_logs",
    log_freq=10000, 
    verbose=1
)

# Train the model with the callback
model.learn(
    total_timesteps=500000, 
    callback=image_logging_callback
)


model.save("./runs/ImgPPO_"+envName+"_sb3_simplifiedv3/"+envName+"_Img_ppo_reach_simplifiedv3")
