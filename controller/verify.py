import pygame
import time
pygame.init()
pygame.joystick.init()

pad = pygame.joystick.Joystick(0)
pad.init()

while True:
    pygame.event.pump()

    for i in range(pad.get_numbuttons()):
        if pad.get_button(i):
            print("Button pressed:", i)
    
    time.sleep(1)