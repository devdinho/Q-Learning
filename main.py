import numpy as np
import gymnasium as gym

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="ansi")

alpha = 0.5         # Taxa de aprendizado
gamma = 0.9         # Fator de desconto
epsilon = 1.0       # Probabilidade inicial de exploração
epsilon_decay = 0.99  # Taxa de decaimento do epsilon
epsilon_min = 0.01  # Valor mínimo de epsilon
num_episodes = 5000
max_steps = 100

state_space_size = env.observation_space.n
action_space_size = env.action_space.n
Q = np.zeros((state_space_size, action_space_size))

def train():
    global epsilon
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()[0]
        total_reward = 0

        for _ in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, _, _ = env.step(action)

            best_next_action = np.argmax(Q[next_state, :])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

            state = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        rewards.append(total_reward)

        if (episode + 1) % 500 == 0:
            print(f"Episódio {episode + 1}, Recompensa Total Média: {np.mean(rewards[-500:]):.2f}")

    return rewards

rewards = train()

print("\nTabela Q Final:")
print(Q)

def test_agent(episodes=10):
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0

        print(f"\nEpisódio {episode + 1}:")
        for _ in range(max_steps):
            action = np.argmax(Q[state, :])
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            print(env.render())
            state = next_state

            if done:
                break

        print(f"Recompensa Total: {total_reward}")

test_agent()