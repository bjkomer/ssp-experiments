class GoalFindingAgent(object):

    def __init__(self, cleanup_network, localization_network, policy_network):

        self.cleanup_network = cleanup_network
        self.localization_network = localization_network
        self.policy_network = policy_network

        # need a reasonable default for the agent ssp
        # maybe use a snapshot localization network that only uses distance sensors to initialize it?
        self.agent_ssp = '??'

    def act(self, distances, velocity, semantic_goal, map_id, item_memory):

        noisy_goal_ssp = item_memory *~ semantic_goal
        goal_ssp = self.cleanup_network(noisy_goal_ssp.v)

        # Just one step of the recurrent network
        self.agent_ssp = self.localization_network(self.agent_ssp, distances, velocity)

        vel_action = self.policy_network(self.agent_ssp, goal_ssp, map_id)

        # TODO: possibly do a transform on the action output if the environment needs it

        return vel_action
