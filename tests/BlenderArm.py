import pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected by Pygame!")
else:
    controller = pygame.joystick.Joystick(0)
    controller.init()
    print(f"Connected: {controller.get_name()}")
    print(f"Axes: {controller.get_numaxes()}, Buttons: {controller.get_numbuttons()}")
    print("Move your sticks and press buttons (Ctrl+C to exit)...")

    try:
        while True:
            pygame.event.pump()
            axes = [controller.get_axis(i) for i in range(controller.get_numaxes())]
            print(f"Axes values: {[round(a, 2) for a in axes]}", end="\r")
    except KeyboardInterrupt:
        print("\nExited.")

pygame.quit()
