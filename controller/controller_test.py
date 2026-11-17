
import pygame
import time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected!")
    quit()

controller = pygame.joystick.Joystick(0)
controller.init()

print("Controller:", controller.get_name())
print("Axes:", controller.get_numaxes())
print("Buttons:", controller.get_numbuttons())

try:
    while True:
        pygame.event.pump()

        axes = [
            round(controller.get_axis(i), 2)
            for i in range(controller.get_numaxes())
        ]

        buttons = [
            controller.get_button(i)
            for i in range(controller.get_numbuttons())
        ]

        print("Axes:", axes, "| Buttons:", buttons)

        time.sleep(1)

except KeyboardInterrupt:
    print("\nController test stopped.")

finally:
    pygame.quit()
