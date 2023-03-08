from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.algorithms import random_agent


if __name__ == '__main__':
    # Create the environment
    env = rl_environment.Environment("tic_tac_toe")
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]

    # Create the agents
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # Train the Q-learning agents in self-play.
    for cur_episode in range(25000):
        if cur_episode % 1000 == 0:
            print(f"Episodes: {cur_episode}")
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])
        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)
    print("Done!")

    # Evaluate the Q-learning agent against a random agent.
    eval_agents = [agents[0], random_agent.RandomAgent(1, num_actions, "Entropy Master 2000")]
    # Play against each other
    # eval_agents = [agents[0], agents[1]]

    time_step = env.reset()
    while not time_step.last():
        print("")
        print(env.get_state)
        player_id = time_step.observations["current_player"]
        # Note the evaluation flag. A Q-learner will set epsilon=0 here.
        agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
        print(f"Agent {player_id} chooses {env.get_state.action_to_string(agent_output.action)}")
        time_step = env.step([agent_output.action])

    print("")
    print(env.get_state)
    print(time_step.rewards)
