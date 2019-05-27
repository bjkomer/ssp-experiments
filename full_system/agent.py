import torch


class GoalFindingAgent(object):

    def __init__(self, cleanup_network, localization_network, policy_network, snapshot_localization_network):

        self.cleanup_network = cleanup_network
        self.localization_network = localization_network
        self.policy_network = policy_network
        self.snapshot_localization_network = snapshot_localization_network

        # need a reasonable default for the agent ssp
        # maybe use a snapshot localization network that only uses distance sensors to initialize it?
        self.agent_ssp = None

    def snapshot_localize(self, distances, map_id):

        # inputs = torch.cat([torch.Tensor(distances), torch.Tensor(map_id)])
        inputs = torch.cat([distances, map_id], dim=1)  # dim is set to 1 because there is a batch dimension
        self.agent_ssp = self.snapshot_localization_network(inputs)

    def act(self, distances, velocity, semantic_goal, map_id, item_memory):
        """
        :param distances: Series of distance sensor measurements
        :param velocity: 2D velocity of the agent from last timestep
        :param semantic_goal: A semantic pointer for the item of interest
        :param map_id: Vector acting as context for the current map. Typically a one-hot vector
        :param item_memory: Spatial semantic pointer containing a summation of LOC*ITEM
        :return: 2D holonomic velocity action
        """

        noisy_goal_ssp = item_memory *~ semantic_goal
        goal_ssp = self.cleanup_network(torch.Tensor(noisy_goal_ssp.v).unsqueeze(0))

        # Just one step of the recurrent network
        # dim is set to 1 because there is a batch dimension
        # inputs are wrapped as a tuple because the network is expecting a tuple of tensors
        self.agent_ssp = self.localization_network(
            inputs=(torch.cat([velocity, distances, map_id], dim=1),),
            initial_ssp=self.agent_ssp
        ).squeeze(0)

        vel_action = self.policy_network(torch.cat([map_id, self.agent_ssp, goal_ssp], dim=1))

        # TODO: possibly do a transform on the action output if the environment needs it

        return vel_action.squeeze(0).detach().numpy()
