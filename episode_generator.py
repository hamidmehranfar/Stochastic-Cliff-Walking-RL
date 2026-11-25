import numpy as np


def generate_episode(policy, env, render=False):
    states = []
    actions = []
    rewards = []

    current_state = env.reset()[0]

    episode_end = False

    if render:
        env.render()

    while not episode_end:
        states.append(current_state)

        action_probs = policy(current_state)
        selected_action = np.random.choice(len(action_probs), p=action_probs)

        next_state, reward, terminated, truncated, prob = env.step(selected_action)

        episode_end = terminated or truncated

        actions.append(selected_action)
        rewards.append(reward)

        current_state = next_state

        if render:
            env.render()

    return states, actions, rewards
