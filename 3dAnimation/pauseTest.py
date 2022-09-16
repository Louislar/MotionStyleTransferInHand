'''
copy from: https://matplotlib.org/stable/gallery/animation/pause_resume.html
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class PauseAnimation:
    def __init__(self):
        fig, ax = plt.subplots()
        ax.set_title('Click to pause/resume the animation')
        x = np.linspace(-0.1, 0.1, 1000)

        # Start with a normal distribution
        self.n0 = (1.0 / ((4 * np.pi * 2e-4 * 0.1) ** 0.5)
                   * np.exp(-x ** 2 / (4 * 2e-4 * 0.1)))
        self.p, = ax.plot(x, self.n0)

        self.animation = animation.FuncAnimation(
            fig, self.update, frames=200, interval=50, blit=True)
        self.paused = False

        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        fig.canvas.mpl_connect('key_press_event', self.toggle_key_pressed)

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused
    
    def toggle_key_pressed(self, event):
        print(event.key)    # Which key is pressed
        print(event.xdata)
        print(event.ydata)
        if event.key == 'a':
            print('key pressed!!')


    def update(self, i):
        self.n0 += i / 100 % 5
        self.p.set_ydata(self.n0 % 20)
        return (self.p,)


pa = PauseAnimation()
plt.show()