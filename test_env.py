import diambra.arena

env = diambra.arena.make("sfiii3n", render_mode="human")
obs, info = env.reset()

terminated, truncated = False, False

for i in range(100):
    actions = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        break

env.close()
print("✅ Step 1 COMPLETE! Ready for Step 2!")

