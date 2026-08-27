
import os
import torch
import numpy as np
from stable_baselines3 import TD3,SAC
from stable_baselines3.td3.policies import MultiInputPolicy
#from stable_baselines3.her import HerReplayBuffer
from custom_her_replay_buffer import CustomHerReplayBuffer
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from torch import nn
from simulationEnv import SimulationEnv
from realEnv import RealEnv
import subprocess
import signal
import time
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import math

class SaveOnBestEfficiencyCallback(BaseCallback):
    """
    Callback for saving a model that achieves the best 'efficiency score',
    defined as mean_reward / mean_episode_length.

    This callback prioritizes models that solve the task efficiently (high
    reward in few steps). It addresses the trade-off between reward and length.

    :param eval_env: The environment used for evaluation.
    :param n_eval_episodes: The number of episodes to test the agent.
    :param eval_freq: Evaluate the agent every `eval_freq` call of the callback.
    :param log_path: Path to a folder where the evaluations will be saved.
    :param best_model_save_path: Path to a folder where the best model will be saved.
    :param deterministic: Whether the evaluation should use stochastic or deterministic actions.
    :param verbose: Verbosity level.
    """
    def __init__(self, eval_env, n_eval_episodes: int, eval_freq: int, log_path: str, best_model_save_path: str, deterministic: bool = True, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        
        # Initialize the best score to a very low value
        self.best_efficiency_score = -np.inf
        
        # Create save paths if they don't exist
        os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the policy
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=False,
            )

            mean_reward = np.mean(episode_rewards)
            mean_ep_length = np.mean(episode_lengths)

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f}, "
                      f"episode_length={mean_ep_length:.2f}")

            # --- EFFICIENCY CALCULATION AND SAVING LOGIC ---

            # Rule: Only consider valid episodes (length > 1) to avoid division by zero
            # or saving based on broken episodes.
            if mean_ep_length <= 1:
                if self.verbose > 0:
                    print(f"Skipping best model check: mean episode length ({mean_ep_length:.2f}) is <= 1.")
                return True

            # Calculate the efficiency score
            current_efficiency_score = mean_reward / mean_ep_length
            
            if self.verbose > 0:
                print(f"Current efficiency score: {current_efficiency_score:.4f} (Best: {self.best_efficiency_score:.4f})")

            # Log the scores
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/efficiency_score", current_efficiency_score)
            self.logger.dump(self.num_timesteps)

            # Check if the current model has a better efficiency score
            if current_efficiency_score > self.best_efficiency_score:
                self.best_efficiency_score = current_efficiency_score
                if self.verbose > 0:
                    print(f"New best efficiency score: {self.best_efficiency_score:.4f}")
                    print(f"Saving new best model to {self.best_model_save_path}")
                
                # Save the new best model
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                # Note: You can also save the replay buffer if needed, similar to EvalCallback
                # self.model.save_replay_buffer(...)
        
        return True
    
class ActionCheckPolicy(MultiInputPolicy):
    def _predict(self, obs, deterministic=False):
        actions = super()._predict(obs, deterministic)
        
        # Convert to CPU numpy array first
        actions_np = actions.cpu().detach().numpy()
        
        if actions_np.shape[0] > 1:
            action_equal = np.allclose(actions_np[0], actions_np[1], atol=1e-4)
            if action_equal:
                print(f"Actions are identical: {actions_np}")
            else:
                print(f"Actions differ: Env1: {actions_np[0]}, Env2: {actions_np[1]}")
        
        return actions

def _worker(
    remote,
    parent_remote,
    env_fn_wrapper,
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped
    from typing import Any, Callable, Optional, Union
    from stable_baselines3.common.vec_env.patch_gym import _patch_env

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    #remote.send(None) # ready signal from environment
    reset_info: Optional[dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except (EOFError, KeyboardInterrupt):
            env.close()
            remote.close()
            break
        except ConnectionResetError:
            pass  # Parent process died unexpectedly

class SequentialSubprocVecEnv(SubprocVecEnv):
    """SubprocVecEnv that initializes environments sequentially"""
    from typing import Any, Callable, Optional, Union
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: Optional[str] = None):
        import multiprocessing as mp
        from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.env_fns = env_fns
        

        if start_method is None:
            start_method = "spawn"  # Use spawn for better isolation
        ctx = mp.get_context(start_method)

        self.start_method = start_method # Just add for debugging

        self.remotes = []
        self.work_remotes = []
        self.processes = []

        # Initialize environments sequentially
        for i, env_fn in enumerate(env_fns):
            work_remote, remote = ctx.Pipe()
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            
            # Wait for readiness signal
            #work_remote.recv()  # Block until environment sends "ready"
            #remote.recv() # Ready signal from the environment
            time.sleep(15+i)  # Give some time for the environment to initialize
            work_remote.close()

            self.remotes.append(remote)
            self.work_remotes.append(work_remote)
            self.processes.append(process)

        # Get observation/action spaces from first environment
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super(SubprocVecEnv, self).__init__(n_envs, observation_space, action_space)

class WebotsManager:
    def __init__(self, env_id, use_real_arm = False, use_camera = False, use_foundation = False, use_webot = False):
        self.env_id = env_id
        self.proc = None
        self.webots_port = 1234 + env_id
        self.use_real_arm = use_real_arm
        self.use_camera = use_camera
        self.use_foundation = use_foundation
        self.use_webot = use_webot
        
    def __enter__(self):
        env = os.environ.copy()

        # Launch Webots in a subprocess vglrun -d $DISPLAY 
        bash_cmd = []
        bash_cmd.append("export DISPLAY")
        bash_cmd.append(f"export ROS_DOMAIN_ID={self.env_id}")
        bash_cmd.append("source ~/Desktop/igus_rebel/install/setup.bash")
        bash_cmd.append("source ~/Desktop/webot_ros/install/setup.bash")
        bash_cmd.append("source ~/Desktop/webot_ros/.venv/bin/activate")
        if self.use_real_arm:
            bash_cmd.append(f"ros2 launch rebel_webot rebel.launch.py use_rviz:=true webots_port:={self.webots_port} env_id:={self.env_id}")
        if self.use_camera:
            bash_cmd.append("source ~/vietd/azure_kinect/install/setup.bash")
            bash_cmd.append("ros2 launch azure_kinect_ros_driver driver.launch.py")
        if self.use_foundation:
            bash_cmd.append("deactivate ")
            bash_cmd.append("source ~/miniconda3/bin/activate")
            bash_cmd.append("conda activate foundationpose_ros")
            bash_cmd.append("python3 ~/vietd/FoundationPose_ros/run_live.py")
        if self.use_webot:
            bash_cmd.append(f"ros2 launch rebel_webot complete.launch.py webots_port:={self.webots_port} env_id:={self.env_id} >log.txt 2>&1")
        
        final = " && ".join(bash_cmd)
        #final_with_logging = f"({final}) > log.txt 2>&1"
        cmd = ['gnome-terminal', '--', "bash", "-c", final]
    
        self.proc = subprocess.Popen(cmd, preexec_fn=os.setsid,env=env, shell=False)

        print(f"Webots launched with PID: {self.proc.pid}")
        time.sleep(1)  # Wait for Webots to initialize
        return self
    
    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            if self.use_camera:
                os.system("pkill -f driver.launch.py")
            if self.use_foundation:
                os.system("pkill -f run_live.py")
            self.proc.wait(timeout=10)
            

def make_env(env_id, seed=0, use_real_arm=False):
    def _init():
        if use_real_arm:
            manager_real_arm = WebotsManager(env_id, use_real_arm=True)
            manager_camera = WebotsManager(env_id, use_camera=True)
            manager_foundation_pose = WebotsManager(env_id, use_foundation=True)
            manager_real_arm.__enter__()
            #manager_camera.__enter__()
            #manager_foundation_pose.__enter__()
            env = RealEnv(seed=seed + env_id, port=manager_real_arm.webots_port, env_id=env_id)
        else:
            manager = WebotsManager(env_id, use_webot=True)
            manager.__enter__()
            env = SimulationEnv(seed=seed + env_id, port=manager.webots_port, env_id=env_id)

        # Ensure the environment's close method stops the manager
        #original_close = env.close if hasattr(env, 'close') else lambda: None
        def wrapped_close():
            if use_real_arm:
                manager_real_arm.stop()
                manager_camera.stop()
                manager_foundation_pose.stop()
            else:
                manager.stop()
            
        env.close = wrapped_close

        return Monitor(env)
    return _init

def lr_schedule(progress_remaining: float) -> float:
    """
    Custom learning rate schedule with warm-up and cosine decay
    progress_remaining: 1 = start of training, 0 = end of training
    """
    # Configuration (adjust these as needed)
    peak_lr = 1e-3      # Maximum LR after warm-up #1e-3
    min_lr = 1e-6       # Minimum LR at end of training
    warmup_frac = 0.1   # First 10% of training for warm-up
    
    # Current progress through training (0 to 1)
    progress = 1 - progress_remaining
    
    # Warm-up phase: linearly increase from 10% of peak to peak LR
    if progress < warmup_frac:
        warmup_progress = progress / warmup_frac
        return peak_lr * (0.1 + 0.9 * warmup_progress)
    
    # Cosine decay phase: smoothly decrease from peak to min LR
    decay_progress = (progress - warmup_frac) / (1 - warmup_frac)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
    return min_lr + cosine_decay * (peak_lr - min_lr)

def train_model():
    # Configuration
    SEED = 0
    EPISODE_MAX_STEPS = 150
    BATCH_SIZE = 1024 
    BUFFER_SIZE = 10000000
    TIMESTEPS = 10000000#BUFFER_SIZE * 10
    EVAL_FREQ = 5000
    N_ENVS =   1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

     # --- Hyperparameters for a NEW model ---
    INITIAL_HYPERPARAMS = {
        "learning_rate": 3e-4,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_starts": 10000,
        "train_freq": (1, "step"), # Common default: train after every step
        "gradient_steps": 1,      # Common default
        "policy_kwargs": {
            "activation_fn": nn.ReLU,
            "net_arch": {"pi": [256, 256, 256], "qf": [256,256,256]},
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {"weight_decay": 1e-5}
        }
    }

    # --- Hyperparameters to APPLY to a LOADED model ---
    NEW_HYPERPARAMS_FOR_LOADED_MODEL = {
        "learning_rate": 1e-5,  # Fine-tune with a much smaller LR
        "batch_size": 1024,      # Maybe use a larger batch for stable gradients
        "gradient_steps": 1,    # Adjust gradient steps
    }

    # Create vectorized environments
    env_fns = [make_env(i) for i in range(N_ENVS)]
    vec_env = SequentialSubprocVecEnv(env_fns, start_method="spawn") 

    action_noise = NormalActionNoise(
        mean=np.zeros(vec_env.action_space.shape[-1]),
        sigma=0.1 * np.ones(vec_env.action_space.shape[-1])
    )
 
     # Define model path
    model_path = "./sac_stack_best/best_model" 
    
    # Check if model exists and load it
    if os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}")
        model = SAC.load(model_path, env=vec_env, device=DEVICE)
        model.lr_schedule = lr_schedule
        model.batch_size =  NEW_HYPERPARAMS_FOR_LOADED_MODEL["batch_size"]
        model.gradient_steps = NEW_HYPERPARAMS_FOR_LOADED_MODEL["gradient_steps"]
        # model.action_noise = action_noise
        
        if os.path.exists(model_path + "_replay_buffer.pkl"):
            model.load_replay_buffer(model_path + "_replay_buffer")
            
        reset_num_timesteps = True  # Continue training from loaded timesteps
    else:
        
        model = SAC(
            "MultiInputPolicy",
            vec_env,
            # replay_buffer_class=CustomHerReplayBuffer,
            # replay_buffer_kwargs=dict(n_sampled_goal=1, goal_selection_strategy="final"),
            learning_rate=lr_schedule,
            buffer_size=INITIAL_HYPERPARAMS["buffer_size"],
            batch_size=INITIAL_HYPERPARAMS["batch_size"],
            learning_starts=INITIAL_HYPERPARAMS["learning_starts"],
            train_freq=INITIAL_HYPERPARAMS["train_freq"],
            gradient_steps=INITIAL_HYPERPARAMS["gradient_steps"],
            policy_kwargs=INITIAL_HYPERPARAMS["policy_kwargs"],
            #action_noise=action_noise,
            verbose=2,
            seed=SEED,
            device=DEVICE,
            tensorboard_log="./sac_stack_tensorboard/"
        )
        reset_num_timesteps = True  # Start new training
    
    # Callbacks
    efficiency_callback = SaveOnBestEfficiencyCallback(
        eval_env=vec_env,  # Use the same vectorized env for evaluation
        n_eval_episodes=5,
        eval_freq=EVAL_FREQ,
        log_path="./sac_stack_logs/",
        best_model_save_path="./sac_stack_best/",
        deterministic=True,
        verbose=1
    )

    callbacks = [
        CheckpointCallback(
            save_freq=EVAL_FREQ,
            save_path="./sac_stack_models/",
            name_prefix="t3d_reach"
        ),
        EvalCallback(
            vec_env,
            best_model_save_path="./sac_stack_best/",
            log_path="./sac_stack_logs/",
            eval_freq=EVAL_FREQ,
            deterministic=True,
            n_eval_episodes=20
        )
    ]

    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps
        )

    finally:
        # Cleanup
        model.save("sac_stack_final")
        model.save_replay_buffer("sac_stack_final_replay_buffer")
        vec_env.close()
        
        # Terminate any remaining Webots processes
        os.system("pkill -f Webots")

def test_model(n_eval_episodes=10): # Use a different seed for testing
    EPISODE_MAX_STEPS = 150
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    code_dir = os.path.dirname(os.path.abspath(__file__))
    
    best_model_path = os.path.join(code_dir, "sac_stack_best", "best_model2.zip")

    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}")
        print("Please ensure training has run and saved a best model.")
        return

    print(f"Loading best model from: {best_model_path}")

    test_env = SequentialSubprocVecEnv([make_env(0)])  # Create a new environment for testing

    # Load the model
    # The environment is needed for initialization, but SB3 handles observation/action spaces
    model = TD3.load(best_model_path, env=test_env, device= DEVICE)
    print("Model loaded successfully.")

    print(f"Evaluating model for {n_eval_episodes} episodes...")
    # Use evaluate_policy: Set return_episode_rewards=True to get lengths
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        model.get_env(), # Use the environment associated with the loaded model
        n_eval_episodes=n_eval_episodes,
        deterministic=True, # Evaluate deterministically for consistent results
        return_episode_rewards=True # <<< Get rewards AND lengths
    )

    print(f"--- Evaluation Results ---")

    if episode_lengths: # Check if any episodes were actually run
        # Calculate reward stats
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        # Calculate success rate based on episode length
        successful_episodes = sum(1 for length in episode_lengths if length < EPISODE_MAX_STEPS and length > 1)
        # Note: Using strict '<'. Change to '<=' if max steps is inclusive for success.
        total_episodes = len(episode_lengths)
        success_rate = successful_episodes / total_episodes

        print(f"Episodes evaluated: {total_episodes}")
        print(f"Mean reward:        {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Successful episodes: {successful_episodes} (completed in < {EPISODE_MAX_STEPS} steps)")
        print(f"Success Rate:       {success_rate:.2%}")
    else:
        print("Warning: No episodes were completed during evaluation.")


    # Cleanup
    test_env.close()
    print("--------------------------")

def test_sync_model(n_eval_episodes=10): # Use a different seed for testing
    EPISODE_MAX_STEPS = 150
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    code_dir = os.path.dirname(os.path.abspath(__file__))
    
    best_model_path = os.path.join(code_dir, "sac_stack_best", "best_model (Copy 6).zip")

    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}")
        print("Please ensure training has run and saved a best model.")
        return

    print(f"Loading best model from: {best_model_path}")

    test_env = SequentialSubprocVecEnv([make_env(env_id=10, use_real_arm=True)])  # Create a new environment for testing

    time.sleep(15)

    # Load the model
    # The environment is needed for initialization, but SB3 handles observation/action spaces
    model = TD3.load(best_model_path, env=test_env, device= DEVICE)
    print("Model loaded successfully.")

    print(f"Evaluating model for {n_eval_episodes} episodes...")
    # Use evaluate_policy: Set return_episode_rewards=True to get lengths
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        model.get_env(), # Use the environment associated with the loaded model
        n_eval_episodes=n_eval_episodes,
        deterministic=True, # Evaluate deterministically for consistent results
        return_episode_rewards=True # <<< Get rewards AND lengths
    )

    print(f"--- Evaluation Results ---")

    if episode_lengths: # Check if any episodes were actually run
        # Calculate reward stats
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        # Calculate success rate based on episode length
        successful_episodes = sum(1 for length in episode_lengths if length < EPISODE_MAX_STEPS)
        # Note: Using strict '<'. Change to '<=' if max steps is inclusive for success.
        total_episodes = len(episode_lengths)
        success_rate = successful_episodes / total_episodes

        print(f"Episodes evaluated: {total_episodes}")
        print(f"Mean reward:        {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Successful episodes: {successful_episodes} (completed in < {EPISODE_MAX_STEPS} steps)")
        print(f"Success Rate:       {success_rate:.2%}")
    else:
        print("Warning: No episodes were completed during evaluation.")


    # Cleanup
    test_env.close()
    print("--------------------------")



if __name__ == '__main__':
    # Set multiprocessing start method
    torch.set_num_threads(1)  # Prevent MKL/NNPACK thread oversubscription
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train or Test a TD3+HER agent.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode to run: "train" or "test" the best model (default: train)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to run during testing (only applicable in test mode)')

    args = parser.parse_args()

    # Use provided seed, otherwise use default SEED for train, SEED+100 for test
    if args.mode == 'train':
        train_model()
    elif args.mode == 'test':
        test_sync_model(n_eval_episodes=args.episodes)