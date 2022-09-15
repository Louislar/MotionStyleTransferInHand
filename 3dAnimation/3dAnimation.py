# 嘗試撰寫一個可以傳入時間序列3d points資料的畫圖class
from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

class VizAnimation:
    def __init__(self, data) -> None:
        self.data = data
        self.timeCount = 0

        fig = plt.figure()
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        ax.set_title('For testing')
        # TODO: 這邊可以創建多組points分別代表不同類別的資料
        #       但是底下update就不知道要怎麼回傳多組points的資料
        # Sol: 額外再開一個animation.FuncAnimation()即可
        # e.g. animation's positions, mapped positions ...
        self.points, = ax.plot(data[0][0], data[0][1], data[0][2])


        self.animation = animation.FuncAnimation(
            fig, self.update, frames=200, interval=50, blit=True)
        self.paused = False

        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        pass
    def toggle_pause(self):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused
        
    def update(self, frameNum):
        self.points.set_data(self.data[frameNum][0], self.data[frameNum][1])
        self.points.set_3d_properties(self.data[frameNum][2])
        pass