import gymnasium as gym
import magic_cartpole


def test_gravity_action():
    try:
        env = gym.make("MagicCartPole")
        for _ in range(100):
            # Create and reset environment
            env.reset()

            # Take a step with changing gravity
            force = 0
            gravity = 2  # Should change gravity

            prev_gravity = env.unwrapped.gravity
            env.step([force, gravity])

            # After taking a step the gravity should change
            assert env.unwrapped.gravity != prev_gravity
        print("✅ Gravity was modified successfully!")
    except:
        print("❌ Gravity did not change! Wizard cannot modify environment.")


def test_gravity_action_with_mask():
    try:
        env = gym.make("MagicCartPole", gravity_mask=True)
        for _ in range(100):

            env.reset()

            # Take a step with changing gravity
            force = 0
            gravity = 2  # Should change gravity

            prev_gravity = env.unwrapped.gravity
            env.step([force, gravity])

            # After taking a step the gravity should change
            assert env.unwrapped.gravity == prev_gravity
        print("✅ Gravity did not change successfully!")

    except:
        print("❌ Gravity has changeg! Wizard could modify environment.")


if __name__ == "__main__":
    print("Running tests...")
    test_gravity_action()
    test_gravity_action_with_mask()
    print("Done running tests.")
