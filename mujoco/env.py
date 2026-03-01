"""Basic env to walk right with the robot"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class G1RightWalkEnv(gym.Env):

    def __init__(self, model_path, max_steps = 1000):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # --- Reset to stand pose ---
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # --- Action space ---
        self.nu = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.nu,),
            dtype=np.float32,
        )

        # --- Observation space ---
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # --- Episode settings ---
        self.max_steps = max_steps
        self.step_count = 0

        self.initial_y = 0.0

        # Store actuator ranges
        self.ctrl_range = self.model.actuator_ctrlrange.copy()

    # --------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        self.step_count = 0
        self.initial_y = self.data.qpos[1]

        return self._get_obs(), {}

    # --------------------------------------------------

    def step(self, action):
        self.step_count += 1

        # Scale action to actuator range
        ctrl_mid = np.mean(self.ctrl_range, axis=1)
        ctrl_half = (self.ctrl_range[:, 1] - self.ctrl_range[:, 0]) / 2.0

        self.data.ctrl[:] = ctrl_mid + action * ctrl_half

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # ----- Reward: movement to the right (+Y) -----
        current_y = self.data.qpos[1]
        reward = current_y - self.initial_y

        # ----- Fall detection -----
        height = self.data.qpos[2]
        terminated = height < 0.4

        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    # --------------------------------------------------

    def _get_obs(self):
        return np.concatenate(
            [self.data.qpos.copy(), self.data.qvel.copy()]
        )

    # --------------------------------------------------

    def render(self):
        pass