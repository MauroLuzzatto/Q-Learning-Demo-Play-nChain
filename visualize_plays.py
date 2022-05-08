import matplotlib.pyplot as plt


class VisualizePlays(object):
    def __init__(self, *agents, alpha=0.8):
        """
        create a visualization method
        Args:
            agents (object): trained q-learning agent
        """
        self.agents = agents
        self.aplpha = alpha
        self.x_label = "#episodes"

    def plot_reward(self):
        """_summary_"""

        plt.title("Reward over time per episode")
        for i, agent in enumerate(self.agents):
            plt.plot(
                agent.rewards,
                c=agent.color,
                label=agent.name,
                linewidth=1,
                linestyle="-",
                alpha=self.aplpha,
            )
        plt.xlabel(self.x_label)
        plt.ylabel("Reward")
        plt.grid()
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=1,
        )

    def plot_epsilon(self):
        """_summary_"""

        # 2
        plt.title("Exploration parameter epsilon per episode")
        for i, agent in enumerate(self.agents):
            plt.plot(
                agent.epsilons,
                c=agent.color,
                label=agent.name,
                linewidth=1,
                linestyle="-",
                alpha=self.aplpha,
            )
        plt.xlabel(self.x_label)
        plt.ylabel("Epsilon")
        plt.grid()
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=1,
        )

    def plot_last_agent_state(self):
        """_summary_"""
        plt.title("Last state the agent is standing on at the end of the episode")
        for i, agent in enumerate(self.agents):
            plt.plot(
                agent.last_states,
                c=agent.color,
                label=agent.name,
                linewidth=1,
                linestyle="-",
                alpha=self.aplpha,
            )
        plt.xlabel(self.x_label)
        plt.ylabel("last state number")
        plt.grid()
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=1,
        )

    def plot_q_values(self):
        """_summary_"""
        plt.title("Average of the value of the q-table")
        for i, agent in enumerate(self.agents):
            plt.plot(
                agent.q_averages,
                c=agent.color,
                label=agent.name,
                linewidth=1,
                linestyle="-",
                alpha=self.aplpha,
            )
        plt.xlabel(self.x_label)
        plt.ylabel("q-table average values")
        plt.grid()
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=1,
        )

    def plot(self):
        """_summary_"""

        plt.figure(figsize=(10, 20))

        plt.subplot(4, 1, 1)
        self.plot_reward()

        plt.subplot(4, 1, 2)
        self.plot_epsilon()

        plt.subplot(4, 1, 3)
        self.plot_last_agent_state()

        plt.subplot(4, 1, 4)
        self.plot_q_values()

        plt.tight_layout()
        plt.show()
