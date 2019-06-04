import torch
import numpy as np
# from spatial_semantic_pointers.utils import encode_point


class GoalFindingAgent(object):

    def __init__(self,
                 cleanup_network, localization_network, policy_network, snapshot_localization_network,
                 cleanup_gt, localization_gt, policy_gt, snapshot_localization_gt,
                 use_snapshot_localization=False,
                 ):

        self.cleanup_network = cleanup_network
        self.localization_network = localization_network
        self.policy_network = policy_network
        self.snapshot_localization_network = snapshot_localization_network

        # Ground truth versions of the above functions
        self.cleanup_gt = cleanup_gt
        self.localization_gt = localization_gt
        self.policy_gt = policy_gt
        self.snapshot_localization_gt = snapshot_localization_gt

        # If true, the snapshot localization network will be used every step
        self.use_snapshot_localization = use_snapshot_localization

        # need a reasonable default for the agent ssp
        # maybe use a snapshot localization network that only uses distance sensors to initialize it?
        self.agent_ssp = None

    def snapshot_localize(self, distances, map_id, env, use_localization_gt=False):

        if use_localization_gt:
            self.agent_ssp = self.localization_gt(env)
        else:
            # inputs = torch.cat([torch.Tensor(distances), torch.Tensor(map_id)])
            inputs = torch.cat([distances, map_id], dim=1)  # dim is set to 1 because there is a batch dimension
            self.agent_ssp = self.snapshot_localization_network(inputs)

    def act(self, distances, velocity, semantic_goal, map_id, item_memory, env,
            use_cleanup_gt=False, use_localization_gt=False, use_policy_gt=False):
        """
        :param distances: Series of distance sensor measurements
        :param velocity: 2D velocity of the agent from last timestep
        :param semantic_goal: A semantic pointer for the item of interest
        :param map_id: Vector acting as context for the current map. Typically a one-hot vector
        :param item_memory: Spatial semantic pointer containing a summation of LOC*ITEM
        :param env: Instance of the environment, for use with ground truth methods
        :param use_cleanup_gt: If set, use a ground truth cleanup memory
        :param use_localization_gt: If set, use the ground truth localization
        :param use_policy_gt: If set, use a ground truth policy to compute the action
        :return: 2D holonomic velocity action
        """

        with torch.no_grad():
            if use_cleanup_gt:
                goal_ssp = self.cleanup_gt(env)
            else:
                noisy_goal_ssp = item_memory *~ semantic_goal
                goal_ssp = self.cleanup_network(torch.Tensor(noisy_goal_ssp.v).unsqueeze(0))
                # Normalize the result
                goal_ssp = goal_ssp / float(np.linalg.norm(goal_ssp.detach().numpy()))

            if use_localization_gt:
                self.agent_ssp = self.localization_gt(env)
            else:
                if self.use_snapshot_localization:
                    self.snapshot_localize(distances=distances, map_id=map_id, env=env, use_localization_gt=False)
                else:
                    # Just one step of the recurrent network
                    # dim is set to 1 because there is a batch dimension
                    # inputs are wrapped as a tuple because the network is expecting a tuple of tensors
                    self.agent_ssp = self.localization_network(
                        inputs=(torch.cat([velocity, distances, map_id], dim=1),),
                        initial_ssp=self.agent_ssp
                    ).squeeze(0)

            if use_policy_gt:
                vel_action = self.policy_gt(map_id=map_id, agent_ssp=self.agent_ssp, goal_ssp=goal_ssp, env=env)
            else:
                vel_action = self.policy_network(
                    torch.cat([map_id, self.agent_ssp, goal_ssp], dim=1)
                ).squeeze(0).detach().numpy()

            # TODO: possibly do a transform on the action output if the environment needs it

            return vel_action

    # def compute_goal_ssp(self, semantic_goal, item_memory, ground_truth=False):
    #     if ground_truth:
    #         pass
    #     else:
    #         noisy_goal_ssp = item_memory * ~ semantic_goal
    #         goal_ssp = self.cleanup_network(torch.Tensor(noisy_goal_ssp.v).unsqueeze(0))
    #     return goal_ssp
    #
    # def compute_agent_ssp(self, velocity, distances, map_id, x_env, y_env, ground_truth=False):
    #     if ground_truth:
    #         # x = ((x_env - 0) / self.coarse_size) * self.limit_range + self.xs[0]
    #         # y = ((y_env - 0) / self.coarse_size) * self.limit_range + self.ys[0]
    #         # agent_ssp = encode_point(x, y, self.x_axis_sp, self.y_axis_sp)
    #         agent_ssp = self.localization_gt(x_env, y_env)
    #     else:
    #         agent_ssp = self.localization_network(
    #             inputs=(torch.cat([velocity, distances, map_id], dim=1),),
    #             initial_ssp=self.agent_ssp
    #         ).squeeze(0)
    #     return agent_ssp
    #
    # def compute_action(self, map_id, agent_ssp, goal_ssp, ground_truth=False):
    #     if ground_truth:
    #         pass
    #     else:
    #         vel_action = self.policy_network(torch.cat([map_id, agent_ssp, goal_ssp], dim=1))
    #         return vel_action.squeeze(0).detach().numpy()
